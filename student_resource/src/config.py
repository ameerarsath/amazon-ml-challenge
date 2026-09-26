"""
Configuration module for Business Entity Resolution.
Contains all paths, constants, and tunable parameters.
"""
import os

# --- Project Paths ---
# Detect if we're running from student_resource or the project root
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_script_dir)  # parent of src/

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- Training Data Files ---
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# --- Test Data Files ---
TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

# --- Output Files ---
MATCHING_RESULTS = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# --- Model Parameters ---
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2  # fraction of S1 entities to hold out for validation

# Blocking parameters
MIN_TOKEN_OVERLAP = 1  # minimum shared tokens for name blocking
TFIDF_TOP_K = 50  # number of top TF-IDF candidates per entity

# Matching threshold (will be tuned on validation data)
MATCH_THRESHOLD = 0.5  # default, will be optimized for F0.5

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
