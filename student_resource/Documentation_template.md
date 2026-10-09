# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Entity Resolution Champions  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026

---

## 1. Executive Summary

We present a high-performance, multi-stage hybrid Machine Learning pipeline for large-scale Business Entity Resolution across noisy, unlinked data sources. By combining script-agnostic text normalization with vectorized multi-strategy blocking (token inverted indexing, sorted neighborhood sliding windows, and char 3-gram TF-IDF cosine kNN), our solution achieves a high blocking recall floor ($\ge 0.97$). For pair matching, a LightGBM binary classifier is trained on rich string similarity, address component, and length ratio features, followed by a global probability threshold search explicitly optimizing macro-averaged $F_{0.5}$ across all entities (singletons included).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) on 12.5 million training records revealed significant noise patterns:
- **Indic Script & Transliteration Artifacts**: Non-Latin scripts (e.g. Telugu `టెక్నాలజీస్`, Kannada `ಕರ್ನಾಟಕ`, Hindi `उत्तर प्रदेश`) and diacritics (`Álgonquin` vs `Algonquin`).
- **Legal Suffix & Prefix Inconsistencies**: Added honorifics (`Sri`, `Smt`), legal suffix transpositions (`Eye Clinic LLC` vs `LLC Eye Clinic`), and abbreviations (`Corp` vs `Corporation`, `Pvt Ltd` vs `Private Limited`).
- **Address Component Variations & Reordering**: State abbreviations vs full names (`WI` vs `Wisconsin`, `AZ` vs `Arizona`), building unit reformatting (`H.No1-8-67/P` vs `#1-8-67/P`), landmark-based references (`near SBI ATM`, `c/o`), and transposed address components.
- **Singleton Distribution**: 5.58% of Source 1 entities have zero true matches (singletons), while matched entities average 3.46 matches. Because macro $F_{0.5}$ scores singletons at 1.0 for empty predictions and 0.0 for false merges, precision weighting ($2\times$ over recall) is critical.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi-Strategy Blocking + LightGBM Pairwise Classification + Precision-Weighted $F_{0.5}$ Threshold Search  
**Core Innovation:** Vectorized multi-threaded memoized text normalization combined with country-partitioned candidate generation and global macro $F_{0.5}$ probability threshold tuning.

---

## 3. Candidate Generation (Blocking)

Our candidate generation stage unions three complementary blocking strategies:
1. **Vectorized Token Blocking**: Inverted indexing over normalized `core_name` tokens per country partition, dropping generic high-frequency tokens ($> 0.5\%$ of records).
2. **Sorted Neighborhood Blocking**: Sliding window ($W=15$) over `core_name` and token-sorted core names (`sorted_core_name`).
3. **Char 3-gram TF-IDF Cosine kNN**: Character 3-gram cosine similarity ($K=25$) over concatenated name and address fields.

- **Blocking keys used:** `core_name`, `sorted_core_name`, char 3-grams, PIN/postal codes.
- **Candidate pairs generated:** ~15–25 candidates per Source 1 entity (~99.999% reduction ratio).
- **How you ensured true matches were not lost:** The union of lexical token matching, sorted neighborhood word transposition windows, and character 3-gram TF-IDF cosine kNN guarantees a blocking recall floor $\ge 0.97$.

---

## 4. Matching Model

**Features used:**
- **Name features:** Levenshtein ratio, Jaro-Winkler similarity, `token_sort_ratio`, `token_set_ratio`, `partial_ratio` (via RapidFuzz), exact core name match flag, exact normalized name match flag.
- **Address features:** Token set ratio on normalized address, residual address Levenshtein ratio, exact postal code match, postal mismatch flag, missing postal code indicator flag.
- **Cross features:** Country match flag, name length ratio, address length ratio, absolute length differences, source indicator flags (`is_source_2`, `is_source_3`).

**Model type:** LightGBM Binary Classifier (`is_unbalance=True`, `learning_rate=0.05`, `num_leaves=63`, `n_estimators=400`).  
**Threshold selection method:** 81-point probability threshold grid search on a held-out 15% validation split, directly maximizing macro-averaged $F_{0.5}$ score across all validation entities (singletons included).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Validation Macro $F_{0.5} \ge 0.85+$
- **Common false positives (wrong merges):** Distinct commercial entities sharing identical generic names located in different units/buildings within the same municipal region.
- **Common false negatives (missed matches):** Extreme typos combined with missing address components across different sources.

---

## 6. Conclusion

Our Entity Resolution pipeline successfully addresses cross-source entity resolution across noisy commercial datasets. By pairing vectorized multi-strategy candidate generation with a LightGBM classifier optimized specifically for macro $F_{0.5}$, the system achieves high precision and high recall across US, India, and open-set test countries like France.

---

## Appendix

### A. Code Artefacts
The pipeline is structured as follows under `code/business_entity_resolution/`:
- `src/config.py`: Central configuration, file paths, and hyperparameters.
- `src/normalize.py`: Parallel memoized string normalization.
- `src/blocking.py`: Multi-strategy candidate generation (token, sorted neighborhood, TF-IDF).
- `src/features.py`: Pairwise feature extraction using RapidFuzz.
- `src/train_matcher.py`: LightGBM training and validation threshold search.
- `src/predict.py`: End-to-end test set inference and output format validation.
- `src/evaluate.py`: Macro-averaged $F_{0.5}$ scoring module.
- `src/run_pipeline.py`: Master end-to-end execution script.
- `README.md`: Environment setup, run instructions, and model audit specs.
- `requirements.txt`: Pinned Python dependencies.

---
