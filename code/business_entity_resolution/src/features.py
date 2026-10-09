import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_pairwise_features(pairs_df: pd.DataFrame, records_df: pd.DataFrame) -> pd.DataFrame:
    """
    pairs_df: DataFrame with columns ['source1_entity_id', 'candidate_entity_id']
    records_df: DataFrame indexed by 'entity_id' with normalized fields:
        ['norm_name', 'core_name', 'sorted_core_name', 'norm_address', 'residual_address', 'postal_code', 'country']
    """
    s1_ids = pairs_df['source1_entity_id'].values
    cand_ids = pairs_df['candidate_entity_id'].values
    
    # Map fields for S1 and Candidates
    s1_rec = records_df.loc[s1_ids]
    cand_rec = records_df.loc[cand_ids]
    
    s1_norm_name = s1_rec['norm_name'].values
    cand_norm_name = cand_rec['norm_name'].values
    
    s1_core_name = s1_rec['core_name'].values
    cand_core_name = cand_rec['core_name'].values
    
    s1_norm_addr = s1_rec['norm_address'].values
    cand_norm_addr = cand_rec['norm_address'].values
    
    s1_res_addr = s1_rec['residual_address'].values
    cand_res_addr = cand_rec['residual_address'].values
    
    s1_postal = s1_rec['postal_code'].values
    cand_postal = cand_rec['postal_code'].values
    
    s1_country = s1_rec['country'].values
    cand_country = cand_rec['country'].values
    
    n_pairs = len(pairs_df)
    features = {}
    
    # 1. RapidFuzz Name Similarities
    lev_ratio = np.zeros(n_pairs, dtype=np.float32)
    jaro_winkler = np.zeros(n_pairs, dtype=np.float32)
    token_sort = np.zeros(n_pairs, dtype=np.float32)
    token_set = np.zeros(n_pairs, dtype=np.float32)
    partial_ratio = np.zeros(n_pairs, dtype=np.float32)
    
    for i in range(n_pairs):
        s1_n = s1_norm_name[i]
        c_n = cand_norm_name[i]
        if s1_n and c_n:
            lev_ratio[i] = fuzz.ratio(s1_n, c_n) / 100.0
            jaro_winkler[i] = distance.JaroWinkler.similarity(s1_n, c_n)
            token_sort[i] = fuzz.token_sort_ratio(s1_n, c_n) / 100.0
            token_set[i] = fuzz.token_set_ratio(s1_n, c_n) / 100.0
            partial_ratio[i] = fuzz.partial_ratio(s1_n, c_n) / 100.0
            
    features['name_lev_ratio'] = lev_ratio
    features['name_jaro_winkler'] = jaro_winkler
    features['name_token_sort'] = token_sort
    features['name_token_set'] = token_set
    features['name_partial_ratio'] = partial_ratio
    
    # 2. Exact Match & Core Name Features
    features['exact_core_name'] = (s1_core_name == cand_core_name).astype(np.float32)
    features['exact_norm_name'] = (s1_norm_name == cand_norm_name).astype(np.float32)
    
    # Core name RapidFuzz
    core_token_set = np.zeros(n_pairs, dtype=np.float32)
    for i in range(n_pairs):
        if s1_core_name[i] and cand_core_name[i]:
            core_token_set[i] = fuzz.token_set_ratio(s1_core_name[i], cand_core_name[i]) / 100.0
    features['core_token_set'] = core_token_set
    
    # 3. Address Similarities
    addr_token_set = np.zeros(n_pairs, dtype=np.float32)
    res_addr_lev = np.zeros(n_pairs, dtype=np.float32)
    
    for i in range(n_pairs):
        s1_a = s1_norm_addr[i]
        c_a = cand_norm_addr[i]
        if s1_a and c_a:
            addr_token_set[i] = fuzz.token_set_ratio(s1_a, c_a) / 100.0
            
        s1_r = s1_res_addr[i]
        c_r = cand_res_addr[i]
        if s1_r and c_r:
            res_addr_lev[i] = fuzz.ratio(s1_r, c_r) / 100.0
            
    features['addr_token_set'] = addr_token_set
    features['res_addr_lev'] = res_addr_lev
    features['exact_norm_addr'] = (s1_norm_addr == cand_norm_addr).astype(np.float32)
    
    # Postal Code Features
    has_p1 = (s1_postal != '')
    has_pc = (cand_postal != '')
    both_postal = has_p1 & has_pc
    
    features['postal_match'] = np.where(both_postal, (s1_postal == cand_postal).astype(np.float32), 0.0)
    features['postal_mismatch'] = np.where(both_postal, (s1_postal != cand_postal).astype(np.float32), 0.0)
    features['postal_missing'] = (~both_postal).astype(np.float32)
    
    # 4. Length & Cross Features
    len_s1_name = np.array([len(x) for x in s1_norm_name], dtype=np.float32)
    len_cand_name = np.array([len(x) for x in cand_norm_name], dtype=np.float32)
    features['name_len_diff'] = np.abs(len_s1_name - len_cand_name)
    features['name_len_ratio'] = np.minimum(len_s1_name, len_cand_name) / np.maximum(1.0, np.maximum(len_s1_name, len_cand_name))
    
    len_s1_addr = np.array([len(x) for x in s1_norm_addr], dtype=np.float32)
    len_cand_addr = np.array([len(x) for x in cand_norm_addr], dtype=np.float32)
    features['addr_len_diff'] = np.abs(len_s1_addr - len_cand_addr)
    features['addr_len_ratio'] = np.minimum(len_s1_addr, len_cand_addr) / np.maximum(1.0, np.maximum(len_s1_addr, len_cand_addr))
    
    # Source Indicator
    is_s2 = np.array([cid.startswith('S2-') for cid in cand_ids], dtype=np.float32)
    features['is_source_2'] = is_s2
    features['is_source_3'] = (1.0 - is_s2)
    
    # Country Match
    features['country_match'] = (s1_country == cand_country).astype(np.float32)
    
    feat_df = pd.DataFrame(features, index=pairs_df.index)
    return feat_df

if __name__ == "__main__":
    print("Testing features.py...")
    sample_pairs = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2'],
        'candidate_entity_id': ['S2-100', 'S3-200']
    })
    sample_records = pd.DataFrame({
        'norm_name': ['baba tech pvt ltd', 'encore cabs pvt ltd'],
        'core_name': ['baba tech', 'encore cabs'],
        'sorted_core_name': ['baba tech', 'cabs encore'],
        'norm_address': ['37 90 squaradeal defence colony hyderabad', '137 road 1 banjara hills hyderabad'],
        'residual_address': ['37 90 squaradeal defence colony hyderabad', '137 road 1 banjara hills hyderabad'],
        'postal_code': ['500081', '500034'],
        'country': ['India', 'India']
    }, index=['S1-1', 'S1-2'])
    
    cand_records = pd.DataFrame({
        'norm_name': ['baba tech pvt ltd', 'sri encore cabs pvt ltd'],
        'core_name': ['baba tech', 'encore cabs'],
        'sorted_core_name': ['baba tech', 'cabs encore'],
        'norm_address': ['37 90 squaradeal defence colony secunderabad', '137 road 1 banjara hills hyderabad'],
        'residual_address': ['37 90 squaradeal defence colony secunderabad', '137 road 1 banjara hills hyderabad'],
        'postal_code': ['500081', '500034'],
        'country': ['India', 'India']
    }, index=['S2-100', 'S3-200'])
    
    all_recs = pd.concat([sample_records, cand_records])
    feats = compute_pairwise_features(sample_pairs, all_recs)
    print("Computed Feature Shape:", feats.shape)
    print(feats.head())
    print("features.py PASSED validation!")
