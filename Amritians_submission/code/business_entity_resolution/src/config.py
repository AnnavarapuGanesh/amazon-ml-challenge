import os

# Base paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
BASE_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
STUDENT_RESOURCE_DIR = os.path.join(BASE_DIR, "student_resource")

# Data paths
TRAIN_S1 = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "train", "train_source1.tsv")
TRAIN_S2 = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "train", "train_source2.tsv")
TRAIN_S3 = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "train", "train_source3.tsv")
TRAIN_GT = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "train", "train_ground_truth.tsv")

TEST_S1 = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2 = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3 = os.path.join(STUDENT_RESOURCE_DIR, "dataset", "test", "test_source3.tsv")

OUTPUT_DIR = os.path.join(STUDENT_RESOURCE_DIR, "output")
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

VALIDATE_SCRIPT = os.path.join(STUDENT_RESOURCE_DIR, "utils", "validate_submission.py")

CACHE_DIR = os.path.join(STUDENT_RESOURCE_DIR, ".cache")

SEED = 42

# Sentence Transformer config
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 128

# Blocking config
KNN_TOP_K = 25
SN_WINDOW_SIZE = 15
MAX_TOKEN_FREQ_RATIO = 0.005
MIN_TOKEN_LEN = 3

# Model training config
VAL_SPLIT_RATIO = 0.15
N_JOBS = -1
