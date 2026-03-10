"""
For constanst all throught the ml folder.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
LABELED_DIR     = DATA_DIR / "labeled"
MODELS_DIR      = DATA_DIR / "models"
NOTEBOOKS_DIR   = ROOT_DIR / "notebooks"

# Auto-create directories if they don't exist
for _dir in [RAW_DIR, LABELED_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw input files — drop your three route files here
# ---------------------------------------------------------------------------

RAW_FILES = {
    "OKC_JFK": RAW_DIR / "OKC_JFK.txt",
    "OKC_ORD": RAW_DIR / "OKC_ORD.txt",
    "OKC_DEN": RAW_DIR / "OKC_DEN.txt",
}

# ---------------------------------------------------------------------------
# Pipeline output files
# ---------------------------------------------------------------------------

MERGED_FILE         = RAW_DIR / "merged_notams.txt"        # output of data_loader.py
LABELED_CSV         = LABELED_DIR / "notams_labeled.csv"    # output of label_generator.py
NLP_CACHE_FILE      = LABELED_DIR / "nlp_cache.json"        # LLM cache so we don't re-score

# ---------------------------------------------------------------------------
# Saved model paths
# ---------------------------------------------------------------------------

BASELINE_MODEL_PATH     = MODELS_DIR / "baseline_lr.pkl"        # Layer 1: TF-IDF + LR
EMBEDDING_MODEL_PATH    = MODELS_DIR / "rf_embeddings.pkl"      # Layer 2: Embeddings + RF
TFIDF_VECTORIZER_PATH   = MODELS_DIR / "tfidf_vectorizer.pkl"   # Fitted TF-IDF vectorizer
ANOMALY_MODEL_PATH      = MODELS_DIR / "isolation_forest.pkl"   # Layer 3: Anomaly detection
EMBEDDINGS_CACHE_PATH   = MODELS_DIR / "embeddings_cache.npy"   # Cached sentence embeddings

# ---------------------------------------------------------------------------
# Label generator (LLM scorer) settings
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL     = "claude-sonnet-4-20250514"
MAX_TOKENS          = 300       # Scores are short JSON — no need for more
API_DELAY_SECONDS   = 0.25      # Pause between API calls to avoid rate limiting
MAX_RETRIES         = 3         # Number of retry attempts on API failure
RETRY_BACKOFF       = 2.0       # Exponential backoff multiplier

# Criticality labels — order matters for sorting (most to least critical)
CRITICALITY_LEVELS  = ["HIGH", "MEDIUM", "LOW", "INFO"]
CRITICALITY_ORDER   = {label: i for i, label in enumerate(CRITICALITY_LEVELS)}

# ---------------------------------------------------------------------------
# TF-IDF settings (Layer 1)
# ---------------------------------------------------------------------------

TFIDF_MAX_FEATURES  = 5000      # Vocabulary size cap
TFIDF_NGRAM_RANGE   = (1, 2)    # Unigrams and bigrams (e.g. "RWY CLSD" as one feature)
TFIDF_MIN_DF        = 2         # Ignore terms that appear in fewer than 2 documents

# ---------------------------------------------------------------------------
# Sentence embedding settings (Layer 2)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight, good for short text
EMBEDDING_DIM        = 384                  # Output dimension for all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Random Forest settings (Layer 2)
# ---------------------------------------------------------------------------

RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = None      # Grow full trees — let RF handle overfitting via bagging
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE     = 42

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

TEST_SIZE           = 0.2       # 80% train, 20% test
RANDOM_STATE        = 42        # Fixed seed for reproducibility across all models

# ---------------------------------------------------------------------------
# Anomaly detection settings (Layer 3)
# ---------------------------------------------------------------------------

ISOLATION_FOREST_CONTAMINATION  = 0.05     # Expect ~5% of NOTAMs to be anomalous
ISOLATION_FOREST_N_ESTIMATORS   = 200
ISOLATION_FOREST_RANDOM_STATE   = 42

# ---------------------------------------------------------------------------
# Route metadata — used to tag NOTAMs with their source route
# ---------------------------------------------------------------------------

ROUTE_METADATA = {
    "OKC_JFK": {"departure": "KOKC", "destination": "KJFK"},
    "OKC_ORD": {"departure": "KOKC", "destination": "KORD"},
    "OKC_DEN": {"departure": "KOKC", "destination": "KDEN"},
}

# ---------------------------------------------------------------------------
# NOTAM block separator — must match what your notam_printer outputs
# ---------------------------------------------------------------------------

NOTAM_SEPARATOR = "=" * 80
