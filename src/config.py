"""
config.py
Single source of truth for file paths, feature lists, and shared constants.
Centralising these removes the duplicated path/feature definitions that were
previously copy-pasted across preprocessing, train_models and pathfinder.
"""
import os

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# The raw CSV may ship under different names; we accept the first that exists.
RAW_CSV_CANDIDATES = [
    os.path.join(DATA_DIR, "railrake.csv"),
    os.path.join(DATA_DIR, "ALL_RAKES_CYCL_FY2526_APR_FEB.csv"),
]

PROC_PKL    = os.path.join(DATA_DIR, "processed.pkl")
GRAPH_PKL   = os.path.join(DATA_DIR, "graph.pkl")
SMAP_PKL    = os.path.join(DATA_DIR, "station_map.pkl")
MODELS_PKL  = os.path.join(MODELS_DIR, "models_bundle.pkl")


def resolve_raw_csv():
    """Return the first raw-CSV path that exists, else raise a clear error."""
    for p in RAW_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No raw CSV found. Place your data file at one of:\n  "
        + "\n  ".join(RAW_CSV_CANDIDATES)
    )


# ── ML feature definitions ────────────────────────────────────────────────────
TARGET = "circuit_speed"

FEATURE_COLS = [
    "ldng_uldg_km", "log_dist_km", "ldng_uldg_hor",
    "ldng_uldg_speed", "same_zone", "dist_bucket",
    "actlwght", "chblwght", "load_units",
    "ldngzone", "uldgzone", "ldngfromdvsn", "uldgdvsn",
    "raketype", "grupcmdt", "ldng_stat", "uldg_state",
]
CAT_COLS = [
    "ldngzone", "uldgzone", "ldngfromdvsn", "uldgdvsn",
    "raketype", "grupcmdt", "ldng_stat", "uldg_state",
]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

# distance-bucket edges; shared by preprocessing AND inference so the model
# never sees a different bucketing at serve time than it did at train time.
DIST_BUCKET_BINS   = [0, 200, 500, 1000, 2000, 99999]
DIST_BUCKET_LABELS = [0, 1, 2, 3, 4]


def dist_bucket(km):
    """Bucket a single distance value using the SAME bins as training."""
    for i in range(len(DIST_BUCKET_BINS) - 1):
        if DIST_BUCKET_BINS[i] < km <= DIST_BUCKET_BINS[i + 1]:
            return float(DIST_BUCKET_LABELS[i])
    return float(DIST_BUCKET_LABELS[-1])
