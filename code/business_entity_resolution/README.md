# Business Entity Resolution — High Performance Scalable Pipeline

This repository contains an end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**. Given business records from 3 independent data sources with noisy and inconsistent fields, the pipeline resolves records across sources to identify identical real-world business entities.

## 1. Architecture Overview

### Stage 1: Pre-processing & Open-Set Normalization (`src/er/preprocess.py`)
- **Transliteration & Text Cleanup**: Unicode NFKD accent normalization, lowercase conversion, punctuation removal, whitespace compaction.
- **Phonetic Encoding**: Fast Soundex phonetic encoding of primary brand tokens to bridge transliteration gaps (e.g. *Chandra* vs. *Chander*, *Laxmi* vs. *Lakshmi*, *Agarwal* vs. *Aggarwal*).
- **Open-Set Legal Form Canonicalization**: Handles US (`inc`, `llc`, `corp`, `ltd`), India (`pvt ltd`, `llp`, `opc`), France (`sarl`, `sas`, `eurl`, `sci`), and an open-set union fallback for unseen countries.
- **Prefix Title Stripping**: Removes honorifics (`sri`, `shree`, `dr`, `m/s`, etc.) to isolate the business brand name.
- **Multi-Pattern Postal Extraction**: Country-specific regex patterns for Indian 6-digit PINs, US 5-digit ZIPs, French 5-digit codes, and generic 4–6 digit codes.
- **Address Landmark Normalization**: Expands standard address abbreviations (`st` → `street`, `rd` → `road`, `opp` → `opposite`, `nr` → `near`, `bldg` → `building`, etc.).

### Stage 2: High-Recall Multi-Key Inverted Index Blocking (`src/er/blocking.py`)
- Multi-key inverted index sharded independently per country partition.
- Features 16 complementary key types:
  - Tokens (`T`), Character 3-grams (`G`), Sorted Name (`N`), Address Tokens (`A`), Street Numbers (`S`)
  - Compound Keys: Postal + Name Token (`PT`), Postal + Street Number (`PS`), Name + Postal (`NP`), Name + Locality (`NL`)
  - Phonetic Keys: Soundex of primary tokens (`PH`), Soundex + Postal/Locality (`PHT`)
  - Anchor Brand Keys: First distinctive word (`ANC`), Anchor + Postal (`ANCP`)
  - Token Bigrams: Adjacent token pairs (`TB`)
- **Document-Frequency Capping & Protection**: Frequent uninformative keys are capped, while compound keys are protected from deletion to preserve >95% recall ceiling.
- **Top-K Retrieval with Score-Ratio Cutoff**: Retains the top candidate pool per Source-1 entity while pruning low-confidence noise ($s < \alpha \cdot s_{\max}$).

### Stage 3: Pairwise & Group Context Feature Engineering (`src/er/features.py`)
- 43 country-agnostic similarity features computed using RapidFuzz in C++ multi-threading:
  - String edit distances: Levenshtein ratio, Token-Sort, Token-Set, Partial Ratio, Jaro-Winkler.
  - Character 3-gram Jaccard, exact match flags, length ratios.
  - Token IDF coverage in both directions, rarest shared/missing token IDFs.
  - Acronym matching and phonetic soundex matching.
  - Legal form and digit agreement (+1, 0, -1).
  - Address token Jaccard, house number overlap/conflict, postal code exact/prefix/conflict matching.
  - Group context features: group size, similarity gap to top candidate, intra-group rank.

### Stage 4: LightGBM Pairwise Matcher & Decision Rule (`src/er/model.py`, `src/er/decision.py`)
- **Classifier**: LightGBM binary classifier (MIT license, < 10 MB, strictly $\le 8\text{B}$ parameters) trained with early stopping on grouped held-out validation.
- **Macro-F0.5 Optimized Gated Rule**:
  - `gate`: Predicts empty if $\max(p) < \text{gate}$ (guarantees 1.0 score on singletons).
  - `pair`: Requires candidate $p \ge \text{pair}$.
  - `rel`: Requires candidate $p \ge \text{rel} \cdot \max(p)$.
- **Target Exclusivity Enforcement**: Enforces the 1-to-1 match constraint by assigning each target record only to its highest-confidence Source-1 entity.

---

## 2. Low Memory Guarantee & Hardware Optimization
- **Strict RAM Bounding**: Memory usage is kept **under 1.5 GB** at all times via country-by-country sharding and chunked streaming inference (`s1_chunk = 20_000`).
- **GPU & Multithreading**: LightGBM and RapidFuzz utilize parallel CPU threads / GPU acceleration without memory blowups.
- **Zero Heavy Dependencies**: No heavy sentence transformers or bloated dense matrices; completely immune to out-of-memory crashes.

---

## 3. Installation & Setup

```bash
pip install -r requirements.txt
```

---

## 4. How to Run the Pipeline End-to-End

Execute from the `src/` directory or project root:
```bash
python src/run_pipeline.py \
    --train-dir ../../student_resource/dataset/train \
    --test-dir ../../student_resource/dataset/test \
    --output-dir ../../output \
    --team-name Amritians
```

### Outputs Produced
- `output/matching_results.tsv`: Final predictions formatted for Portal leaderboard scoring.
- `output/candidate_pairs.tsv`: Exact candidate set generated during blocking.
- `Amritians_submission.zip`: Complete submission package containing code, output TSVs, and filled methodology report.
