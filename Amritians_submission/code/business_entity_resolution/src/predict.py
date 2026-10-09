import os
import sys
import gc
import time
import pandas as pd
import numpy as np
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from normalize import normalize_dataframe
from blocking import generate_all_candidates
from features import compute_pairwise_features

def save_id_list_tsv(output_path: str, id_dict: dict, header_name: str):
    """
    Write tab-separated file with header [source1_entity_id, header_name].
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"source1_entity_id\t{header_name}\n")
        for s1_id in sorted(id_dict.keys()):
            ids = id_dict[s1_id]
            id_str = ",".join(ids) if ids else ""
            f.write(f"{s1_id}\t{id_str}\n")
    print(f"Saved {output_path} ({len(id_dict)} rows).", flush=True)

def generate_test_predictions(model, threshold: float):
    print("=== STARTING INFERENCE ON TEST SET ===", flush=True)
    
    # 1. Load Test Data
    print("Loading test files...", flush=True)
    test_s1 = pd.read_csv(config.TEST_S1, sep="\t")
    test_s2 = pd.read_csv(config.TEST_S2, sep="\t")
    test_s3 = pd.read_csv(config.TEST_S3, sep="\t")
    
    print(f"Test S1 shape: {test_s1.shape}", flush=True)
    print(f"Test S2 shape: {test_s2.shape}", flush=True)
    print(f"Test S3 shape: {test_s3.shape}", flush=True)
    
    # Verify open-set country coverage (France included)
    test_countries = set(test_s1['country'].unique())
    print(f"Test Countries Present: {test_countries}", flush=True)

    # 2. Normalize Test Data
    print("Normalizing test data...", flush=True)
    s1_norm = normalize_dataframe(test_s1, "Test S1")
    s2_norm = normalize_dataframe(test_s2, "Test S2")
    s3_norm = normalize_dataframe(test_s3, "Test S3")

    # 3. Candidate Generation (Blocking)
    print("Running Blocking on Test Data...", flush=True)
    cand_dict = generate_all_candidates(s1_norm, s2_norm, s3_norm)
    
    # Write candidate_pairs.tsv immediately
    print("Writing candidate_pairs.tsv...", flush=True)
    save_id_list_tsv(config.CANDIDATE_PAIRS_PATH, cand_dict, "candidate_entity_ids")

    # 4. Feature Engineering on Test Candidate Pairs
    print("Computing features for test candidate pairs...", flush=True)
    all_recs = pd.concat([s1_norm, s2_norm, s3_norm]).set_index('entity_id')
    
    # Convert cand_dict to DataFrame
    pair_rows = []
    for s1_id, cands in cand_dict.items():
        for cand_id in cands:
            pair_rows.append((s1_id, cand_id))
            
    pairs_df = pd.DataFrame(pair_rows, columns=['source1_entity_id', 'candidate_entity_id'])
    print(f"Total Test Candidate Pairs to score: {len(pairs_df)}", flush=True)
    
    matching_dict = {s1_id: [] for s1_id in test_s1['entity_id']}
    
    if not pairs_df.empty:
        # Score in chunks to optimize memory
        chunk_size = 500000
        n_chunks = int(np.ceil(len(pairs_df) / chunk_size))
        
        for i in range(n_chunks):
            chunk_pairs = pairs_df.iloc[i*chunk_size : (i+1)*chunk_size]
            X_chunk = compute_pairwise_features(chunk_pairs, all_recs)
            probs = model.predict(X_chunk, num_iteration=model.best_iteration)
            
            passed_mask = (probs >= threshold)
            passed_pairs = chunk_pairs[passed_mask]
            
            for s1_id, cand_id in zip(passed_pairs['source1_entity_id'], passed_pairs['candidate_entity_id']):
                matching_dict[s1_id].append(cand_id)
                
            gc.collect()

    # Deduplicate matching dict lists and retain order
    for s1_id in matching_dict:
        seen = set()
        deduped = []
        for cid in matching_dict[s1_id]:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        matching_dict[s1_id] = deduped

    # 5. Write matching_results.tsv
    print("Writing matching_results.tsv...", flush=True)
    save_id_list_tsv(config.MATCHING_RESULTS_PATH, matching_dict, "matched_entity_ids")

    # 6. Validate Output Files using utils/validate_submission.py
    print("\n=== RUNNING SUBMISSION VALIDATION SCRIPT ===", flush=True)
    cmd = [
        sys.executable, config.VALIDATE_SCRIPT,
        "--matching", config.MATCHING_RESULTS_PATH,
        "--candidate", config.CANDIDATE_PAIRS_PATH,
        "--test-dir", os.path.dirname(config.TEST_S1)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Validation Output:")
    print(res.stdout)
    if res.returncode != 0:
        print("Validation ERROR Output:")
        print(res.stderr)
        raise RuntimeError("Submission validation failed! Fix issues before proceeding.")
    else:
        print("Submission validation PASSED cleanly!")

if __name__ == "__main__":
    print("predict.py module loaded successfully.")
