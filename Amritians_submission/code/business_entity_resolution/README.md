# Business Entity Resolution — Reproduction Guide

## Requirements
```bash
pip install -r requirements.txt
```

## Running the End-to-End Pipeline
Execute from this directory:
```bash
python src/run_pipeline.py \
    --train-dir ../../student_resource/dataset/train \
    --test-dir  ../../student_resource/dataset/test \
    --output-dir ../../output
```

## Pipeline Architecture
1. **Normalization (`er/preprocess.py`)**: NFKD transliteration, Soundex phonetic encoding, legal form canonicalization, multi-pattern postal extraction.
2. **Blocking (`er/blocking.py`)**: Multi-key inverted index with IDF weights, compound phonetic/locality keys, document-frequency caps. Writes `output/candidate_pairs.tsv`.
3. **Feature Engineering (`er/features.py`)**: 43 RapidFuzz C++ pairwise & group context features.
4. **Matcher Model (`er/model.py`)**: LightGBM binary classifier with early stopping.
5. **Decision Rule (`er/decision.py`)**: 3-parameter macro F0.5 tuned gate/pair/relative rule with target exclusivity. Writes `output/matching_results.tsv`.
6. **Validation (`er/validate.py`)**: Programmatically verifies all submission constraints and runs the official `utils/validate_submission.py`.

Memory footprint is strictly bounded to < 1.5 GB via streaming country partitions.
