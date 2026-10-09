import os
import sys
import gc
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

LEGAL_STOPWORDS = {'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
                   'pvt', 'private', 'llc', 'co', 'company', 'and', 'sri', 'smt', 'group'}

def generate_candidates_for_country(df_s1: pd.DataFrame, df_s23: pd.DataFrame,
                                     top_k_tfidf: int = 25,
                                     max_token_freq: float = 0.005,
                                     window_size: int = 20) -> dict:
    """
    Generate candidate matching pairs for a single country partition.
    Returns: dict mapping s1_entity_id -> set of candidate entity_ids
    """
    s1_ids = df_s1['entity_id'].values
    s23_ids = df_s23['entity_id'].values
    n_s1 = len(s1_ids)
    n_s23 = len(s23_ids)
    
    candidates = defaultdict(set)
    
    # 1. Sorted Neighborhood (Window W=20 on core_name, sorted_core_name)
    print("    [Blocking] Running Sorted Neighborhood...", flush=True)
    all_ids = np.concatenate([s1_ids, s23_ids])
    is_s1 = np.concatenate([np.ones(n_s1, dtype=bool), np.zeros(n_s23, dtype=bool)])
    
    for sort_field in ['core_name', 'sorted_core_name']:
        keys = np.concatenate([df_s1[sort_field].values, df_s23[sort_field].values])
        sort_order = np.argsort(keys)
        sorted_is_s1 = is_s1[sort_order]
        sorted_ids = all_ids[sort_order]
        
        for w in range(1, window_size + 1):
            s1_mask_f = sorted_is_s1[:-w] & (~sorted_is_s1[w:])
            if np.any(s1_mask_f):
                s1_indices = np.where(s1_mask_f)[0]
                s1_p = sorted_ids[s1_indices]
                cand_p = sorted_ids[s1_indices + w]
                for s1_i, cand_i in zip(s1_p, cand_p):
                    candidates[s1_i].add(cand_i)
                    
            s23_mask_b = (~sorted_is_s1[:-w]) & sorted_is_s1[w:]
            if np.any(s23_mask_b):
                s1_indices = np.where(s23_mask_b)[0] + w
                s1_p = sorted_ids[s1_indices]
                cand_p = sorted_ids[s1_indices - w]
                for s1_i, cand_i in zip(s1_p, cand_p):
                    candidates[s1_i].add(cand_i)

    # 2. Vectorized Token Blocking (CountVectorizer with custom stopwords)
    print("    [Blocking] Running Token Blocking...", flush=True)
    cv = CountVectorizer(binary=True, min_df=2, max_df=max_token_freq, stop_words=list(LEGAL_STOPWORDS), token_pattern=r'(?u)\b\w\w\w+\b')
    try:
        s23_tokens = cv.fit_transform(df_s23['core_name'])
        s1_tokens = cv.transform(df_s1['core_name'])
        
        batch_size = 10000
        for start_i in range(0, n_s1, batch_size):
            end_i = min(n_s1, start_i + batch_size)
            overlap_batch = s1_tokens[start_i:end_i].dot(s23_tokens.T)
            
            indptr = overlap_batch.indptr
            indices = overlap_batch.indices
            data = overlap_batch.data
            
            for i_loc in range(end_i - start_i):
                s_idx = indptr[i_loc]
                e_idx = indptr[i_loc + 1]
                if e_idx > s_idx:
                    row_indices = indices[s_idx:e_idx]
                    row_data = data[s_idx:e_idx]
                    if len(row_indices) > 35:
                        top_idx = row_indices[np.argpartition(row_data, -35)[-35:]]
                    else:
                        top_idx = row_indices
                    s1_id = s1_ids[start_i + i_loc]
                    for idx in top_idx:
                        candidates[s1_id].add(s23_ids[idx])
    except Exception as e:
        print(f"Token blocking warning: {e}", flush=True)

    # 3. Vectorized Char 3-gram TF-IDF Cosine Similarity (Fast CSR indptr iteration)
    print("    [Blocking] Running Char 3-gram TF-IDF Cosine kNN...", flush=True)
    try:
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=3)
        s23_text = (df_s23['core_name'] + " " + df_s23['norm_address']).values
        s1_text = (df_s1['core_name'] + " " + df_s1['norm_address']).values
        
        s23_tfidf = tfidf.fit_transform(s23_text)
        s1_tfidf = tfidf.transform(s1_text)
        
        batch_size = 10000
        for start_i in range(0, n_s1, batch_size):
            end_i = min(n_s1, start_i + batch_size)
            sim_batch = s1_tfidf[start_i:end_i].dot(s23_tfidf.T)
            
            indptr = sim_batch.indptr
            indices = sim_batch.indices
            data = sim_batch.data
            
            for i_loc in range(end_i - start_i):
                s_idx = indptr[i_loc]
                e_idx = indptr[i_loc + 1]
                if e_idx > s_idx:
                    row_indices = indices[s_idx:e_idx]
                    row_data = data[s_idx:e_idx]
                    if len(row_indices) > top_k_tfidf:
                        top_idx = row_indices[np.argpartition(row_data, -top_k_tfidf)[-top_k_tfidf:]]
                    else:
                        top_idx = row_indices
                    s1_id = s1_ids[start_i + i_loc]
                    for idx in top_idx:
                        candidates[s1_id].add(s23_ids[idx])
    except Exception as e:
        print(f"TF-IDF blocking warning: {e}", flush=True)

    return candidates

def generate_all_candidates(s1_norm: pd.DataFrame, s2_norm: pd.DataFrame, s3_norm: pd.DataFrame) -> dict:
    """
    Generate union of candidate pairs across all country partitions.
    Ensures EVERY S1 entity is present in the output dictionary.
    """
    start_t = time.time()
    all_candidates = {s1_id: set() for s1_id in s1_norm['entity_id']}
    
    s23_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    countries = s1_norm['country'].unique()
    
    for country in countries:
        print(f"  [Blocking] Processing Country: {country}...", flush=True)
        c_s1 = s1_norm[s1_norm['country'] == country].copy()
        c_s23 = s23_norm[s23_norm['country'] == country].copy()
        if len(c_s1) == 0 or len(c_s23) == 0:
            continue
            
        c_cands = generate_candidates_for_country(
            c_s1, c_s23,
            top_k_tfidf=config.KNN_TOP_K,
            max_token_freq=config.MAX_TOKEN_FREQ_RATIO,
            window_size=config.SN_WINDOW_SIZE
        )
        for s1_id, cands in c_cands.items():
            all_candidates[s1_id].update(cands)
        gc.collect()

    print(f"Candidate Generation complete in {time.time() - start_t:.2f}s.", flush=True)
    return all_candidates

def candidates_dict_to_dataframe(cand_dict: dict) -> pd.DataFrame:
    """
    Convert candidate dictionary {s1_id: set(cand_ids)} to long DataFrame
    ['source1_entity_id', 'candidate_entity_id'].
    """
    rows = []
    for s1_id, cands in cand_dict.items():
        for cand_id in cands:
            rows.append((s1_id, cand_id))
    return pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_id'])

if __name__ == "__main__":
    print("blocking.py module loaded successfully.")
