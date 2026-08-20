import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_JSON_DIR = RESULTS_DIR / "json"
PLOTS_DIR = RESULTS_DIR / "plots"
EMISSIONS_DIR = RESULTS_DIR / "emissions"

# Ensure output directories exist
for path in [RESULTS_DIR, RESULTS_JSON_DIR, PLOTS_DIR, EMISSIONS_DIR]:
    os.makedirs(path, exist_ok=True)

# Default Training Configuration
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_SEQ_LEN = 64

HF_TOKEN = os.environ.get("HF_TOKEN")
