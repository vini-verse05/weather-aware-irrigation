"""Central configuration: paths, constants, column-detection rules.

All paths are relative to the project root (no machine-specific paths).
Dataset-specific choices (file, sector, column names, units) can be overridden
without editing code by creating ``config.local.json`` in the project root
(see ``config.local.example.json``). That file is git-ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.json"

MODEL_FILENAME = "best_model.joblib"
METADATA_FILENAME = "model_metadata.json"
COMPARISON_FILENAME = "model_comparison.csv"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"
RESULTS_SUMMARY_FILENAME = "results_summary.md"

# --------------------------------------------------------------------------
# Forecasting problem
# --------------------------------------------------------------------------
FORECAST_HORIZON_HOURS = 24
RANDOM_STATE = 42

# Chronological split fractions (the remainder is the test set).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# Look-back features, in hours. All of them use PAST observations only.
LAG_HOURS = (1, 3, 6, 12, 24)
ROLLING_HOURS = (6, 24)

TARGET_COLUMN = "soil_moisture_t_plus_24h"

# --------------------------------------------------------------------------
# Column roles
# --------------------------------------------------------------------------
REQUIRED_ROLES = ("timestamp", "soil_moisture")
OPTIONAL_ROLES = ("temperature", "humidity", "precipitation", "et0", "irrigation")

# Regex patterns (matched against lower-cased column names) used to *suggest* a
# mapping from dataset columns to logical roles. This is a heuristic: the
# inspection command prints the detected mapping and it can always be
# overridden in config.local.json -> "column_overrides".
ROLE_PATTERNS: dict[str, list[str]] = {
    "timestamp": [r"^ts$", r"^timestamp$", r"^datetime$", r"^date_?time$", r"^time$", r"^date$", r"timestamp", r"datetime", r"time"],
    "soil_moisture": [r"soil_?moisture", r"\bswc\b", r"\bvwc\b", r"moisture", r"^sm(_|\b)"],
    "temperature": [r"temp"],
    "humidity": [r"humid", r"\brh\b", r"^rh"],
    "precipitation": [r"precip", r"rain"],
    "et0": [r"\bet0\b", r"^et0", r"evapotransp", r"\beto\b"],
    "irrigation": [r"irrig", r"applied_?water", r"water_?volume", r"\bvolume\b"],
}

# Columns that look like engineered features or targets of an analysis-ready
# table are never auto-selected as *raw* inputs (prevents target leakage).
EXCLUDED_NAME_PATTERNS = [
    r"target", r"future", r"ahead", r"t\+\d+", r"plus_?\d+", r"next_", r"_next",
    r"lag", r"roll", r"_mean", r"_std", r"_min$", r"_max$", r"diff", r"delta",
    r"_sin$", r"_cos$", r"flag", r"quality", r"imputed",
]

TARGET_LIKE_PATTERNS = [r"target", r"24h.*ahead", r"ahead", r"t\+24", r"plus_?24", r"future", r"next_?24"]

# Physical sanity ranges used only to blank impossible values (units permitting).
HUMIDITY_VALID_RANGE = (0.0, 100.0)

# --------------------------------------------------------------------------
# Weather / location defaults
# --------------------------------------------------------------------------
# The example location is the experimental farm of the primary dataset
# (Arnesano, Lecce, Italy) - approximate coordinates; editable in the dashboard.
DEFAULT_LOCATION = {"name": "Arnesano, Lecce, Italy (dataset site, approx.)", "latitude": 40.33, "longitude": 18.10}

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "et0_fao_evapotranspiration",
    "wind_speed_10m",
)
ARCHIVE_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "et0_fao_evapotranspiration",
    "wind_speed_10m",
)
HTTP_TIMEOUT_SECONDS = (5, 15)  # (connect, read)
HTTP_RETRIES = 2
FORECAST_CACHE_TTL_SECONDS = 30 * 60
ARCHIVE_CACHE_TTL_SECONDS = 24 * 60 * 60

# --------------------------------------------------------------------------
# Irrigation recommendation defaults (NOT agronomically validated)
# --------------------------------------------------------------------------
RAIN_AMOUNT_THRESHOLD_MM = 2.0
RAIN_PROBABILITY_THRESHOLD_PCT = 50.0
RAIN_WINDOW_HOURS = 24

CALIBRATION_NOTICE = (
    "Moisture thresholds are not universal. They depend on crop, soil type, root depth and the "
    "sensor's units/calibration, and must be calibrated for your own field. This tool provides "
    "decision support only and does not trigger any irrigation equipment."
)

DEFAULT_SOIL_MOISTURE_UNIT = "dataset units (confirm in the dataset's DATA_DICTIONARY.md)"


# --------------------------------------------------------------------------
# Local overrides
# --------------------------------------------------------------------------
def load_local_overrides(path: Path | None = None) -> dict[str, Any]:
    """Read ``config.local.json`` if present; return {} otherwise.

    Recognised keys: ``data_file``, ``sector_column``, ``sector_value``,
    ``soil_moisture_unit``, ``column_overrides`` (role -> column name).
    """
    cfg_path = Path(path) if path else LOCAL_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {cfg_path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path.name} must contain a JSON object.")
    return data


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
