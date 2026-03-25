from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEFAULT_DATA_CANDIDATES = [
    RAW_DATA_DIR / "phishing_dataset.csv",
    PROJECT_ROOT / "phishing_dataset.csv",
    PROJECT_ROOT / "Training.parquet",
]

MODEL_PATH = ARTIFACTS_DIR / "hybrid_cnn_model.pt"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"

MAX_URL_LENGTH = 256
RANDOM_STATE = 42
