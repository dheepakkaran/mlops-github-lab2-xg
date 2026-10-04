from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "shots.csv"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "metrics"
REPORTS_DIR = ROOT / "reports"

FEATURES = [
    "distance", "angle", "is_header", "first_time", "under_pressure",
    "one_on_one", "from_free_kick", "from_counter", "from_set_piece",
]
TARGET = "is_goal"
SEED = 42


def load_splits():
    df = pd.read_csv(DATA_PATH)
    train, temp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    calib, test = train_test_split(temp, test_size=0.5, stratify=temp[TARGET], random_state=SEED)
    return train, calib, test


def model_path(version, calibrated=False):
    suffix = "_calibrated" if calibrated else ""
    return MODELS_DIR / f"xg_model_{version}{suffix}.joblib"
