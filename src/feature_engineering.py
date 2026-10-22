"""Feature and target construction for 24-hour-ahead soil-moisture forecasting.

Design rules (to prevent data leakage):
* Every feature uses observations at or BEFORE the prediction timestamp.
* The target is looked up by *timestamp* (t + horizon), not by row position,
  so gaps in the series produce a missing target instead of a wrong one.
* Weather features are past observations from the dataset. Future weather is
  never used as a model input (forecasts are used only by the recommendation
  step, see docs/methodology.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

REQUIRED_FEATURES = ("sm_current", "sm_lag_24h")


def shifted_values(series: pd.Series, offset: pd.Timedelta, tolerance: pd.Timedelta) -> pd.Series:
    """Value of ``series`` at ``index + offset`` (nearest within ``tolerance``).

    Positive offsets look into the future (used ONLY to build the target);
    negative offsets look into the past (lag features).
    """
    if not series.index.is_monotonic_increasing or not series.index.is_unique:
        raise ValueError("Series index must be sorted and unique.")
    looked_up = series.reindex(series.index + offset, method="nearest", tolerance=tolerance)
    return pd.Series(looked_up.to_numpy(), index=series.index)


def add_target(
    df: pd.DataFrame,
    soil_col: str,
    interval: pd.Timedelta,
    horizon_hours: int = config.FORECAST_HORIZON_HOURS,
) -> pd.Series:
    """Soil moisture observed ``horizon_hours`` after each timestamp (NaN if not observed)."""
    return shifted_values(df[soil_col], pd.Timedelta(hours=horizon_hours), interval / 2).rename(config.TARGET_COLUMN)


def build_features(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    interval: pd.Timedelta,
    include_target: bool = True,
    horizon_hours: int = config.FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """Create the model feature table from a cleaned time series.

    Returns a DataFrame indexed by timestamp with feature columns and, when
    ``include_target`` is True, the target column. Rows missing the required
    features (current value, 24 h lag) or the target are dropped.
    """
    sm = mapping["soil_moisture"]
    if sm not in df.columns:
        raise KeyError(f"Soil-moisture column '{sm}' not found in the data.")
    tol = interval / 2
    out = pd.DataFrame(index=df.index)

    s = df[sm]
    out["sm_current"] = s
    for h in config.LAG_HOURS:
        lagged = shifted_values(s, -pd.Timedelta(hours=h), tol)
        out[f"sm_lag_{h}h"] = lagged
        out[f"sm_change_{h}h"] = s - lagged
    for h in config.ROLLING_HOURS:
        out[f"sm_roll_mean_{h}h"] = s.rolling(f"{h}h", min_periods=1).mean()

    for role, name in (("temperature", "temp"), ("humidity", "hum")):
        col = mapping.get(role)
        if col and col in df.columns:
            out[f"{name}_current"] = df[col]
            out[f"{name}_roll_mean_24h"] = df[col].rolling("24h", min_periods=1).mean()

    precip = mapping.get("precipitation")
    if precip and precip in df.columns:
        out["precip_sum_6h"] = df[precip].rolling("6h", min_periods=1).sum()
        out["precip_sum_24h"] = df[precip].rolling("24h", min_periods=1).sum()

    for role, name in (("et0", "et0_sum_24h"), ("irrigation", "irrigation_sum_24h")):
        col = mapping.get(role)
        if col and col in df.columns:
            out[name] = df[col].rolling("24h", min_periods=1).sum()

    hours = out.index.hour + out.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    doy = out.index.dayofyear.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    out = out.dropna(subset=list(REQUIRED_FEATURES))
    if include_target:
        target = add_target(df, sm, interval, horizon_hours).reindex(out.index)
        out[config.TARGET_COLUMN] = target
        out = out.dropna(subset=[config.TARGET_COLUMN])
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature column names (everything except the target), excluding all-empty columns."""
    return [c for c in frame.columns if c != config.TARGET_COLUMN and frame[c].notna().any()]


def chronological_split(
    frame: pd.DataFrame,
    train_frac: float = config.TRAIN_FRACTION,
    val_frac: float = config.VAL_FRACTION,
    horizon_hours: int = config.FORECAST_HORIZON_HOURS,
) -> dict[str, pd.DataFrame]:
    """Split a time-indexed frame into train / validation / test in time order.

    A purge gap of ``horizon_hours`` is removed from the end of the training and
    validation sets: a sample at time t has its target at t + horizon, so
    without the gap, targets of late training rows would be observed during the
    validation period (and likewise for validation/test), which leaks future
    information. No random shuffling is ever used.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Fractions must satisfy 0 < train, val and train + val < 1.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Frame must be sorted chronologically before splitting.")
    n = len(frame)
    if n < 30:
        raise ValueError(f"Too few rows ({n}) to split into train/validation/test.")
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    gap = pd.Timedelta(hours=horizon_hours)
    val_start, test_start = frame.index[i1], frame.index[i2]

    train = frame.iloc[:i1]
    train = train[train.index <= val_start - gap]
    val = frame.iloc[i1:i2]
    val = val[val.index <= test_start - gap]
    test = frame.iloc[i2:]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A split is empty after applying the purge gap; the series is too short.")
    return {"train": train, "val": val, "test": test}
