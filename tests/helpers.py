"""Shared helpers for the test-suite.

SYNTHETIC FIXTURES - these frames are randomly generated purely to exercise the
software (loading, feature building, splitting, training, prediction). Results
obtained from them say NOTHING about real model performance and must never be
reported as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import detect_columns, clean_timeseries, infer_sampling_interval

SYNTHETIC_COLUMNS = {
    "timestamp": "timestamp",
    "soil_moisture": "soil_moisture",
    "temperature": "air_temperature",
    "humidity": "relative_humidity",
    "precipitation": "precipitation_mm",
    "et0": "et0_mm",
    "irrigation": "irrigation_volume",
}


def make_synthetic_frame(days: int = 40, freq: str = "30min", seed: int = 0) -> pd.DataFrame:
    """Random-but-plausible irrigation time series (SYNTHETIC, for tests only)."""
    rng = np.random.default_rng(seed)
    periods = int(days * 86400 / pd.Timedelta(freq).total_seconds())
    idx = pd.date_range("2025-03-01", periods=periods, freq=freq)
    n = len(idx)
    hours = idx.hour + idx.minute / 60
    temp = 18 + 8 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.5, n)
    hum = np.clip(65 - 1.5 * (temp - 18) + rng.normal(0, 3, n), 5, 100)
    rain = np.where(rng.random(n) < 0.01, rng.exponential(2.0, n), 0.0)
    irrig = np.where((idx.hour == 6) & (idx.minute == 0) & (rng.random(n) < 0.5), 5.0, 0.0)
    sm = np.empty(n)
    sm[0] = 30.0
    for i in range(1, n):
        sm[i] = sm[i - 1] - 0.01 * max(temp[i], 0) / 10 + 0.8 * rain[i] + 0.5 * irrig[i] + rng.normal(0, 0.03)
        sm[i] = min(max(sm[i], 5.0), 45.0)
    return pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
        "soil_moisture": sm,
        "air_temperature": temp,
        "relative_humidity": hum,
        "precipitation_mm": rain,
        "et0_mm": np.clip(temp, 0, None) * 0.01,
        "irrigation_volume": irrig,
    })


def synthetic_cleaned(days: int = 40, freq: str = "30min", seed: int = 0):
    """Return (cleaned_df, mapping, interval) built from the synthetic frame."""
    raw = make_synthetic_frame(days, freq, seed)
    mapping, _ = detect_columns(list(raw.columns))
    df, _ = clean_timeseries(raw, mapping)
    interval, _ = infer_sampling_interval(df.index)
    return df, mapping, interval


def write_synthetic_csv(path, days: int = 40, freq: str = "30min", seed: int = 0):
    make_synthetic_frame(days, freq, seed).to_csv(path, index=False)
    return path


_TRAINED_DIR_CACHE: dict[str, str] = {}


def trained_model_dir() -> str:
    """Train a small model ONCE on synthetic data and return the models directory.

    Used only to test loading/prediction code paths (SYNTHETIC - not a real result).
    """
    if "models" not in _TRAINED_DIR_CACHE:
        import tempfile
        from pathlib import Path

        from src import train_model

        root = Path(tempfile.mkdtemp(prefix="irrigation_test_"))
        csv = write_synthetic_csv(root / "synthetic.csv", days=30, freq="1h", seed=1)
        train_model.run(csv, models_dir=root / "models", reports_dir=root / "reports",
                        make_plots=False, overrides={}, verbose=False)
        _TRAINED_DIR_CACHE["models"] = str(root / "models")
        _TRAINED_DIR_CACHE["reports"] = str(root / "reports")
    return _TRAINED_DIR_CACHE["models"]
