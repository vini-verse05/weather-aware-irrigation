"""Time-series target construction, leakage safety and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import add_target, build_features, chronological_split, feature_columns, shifted_values
from tests.helpers import synthetic_cleaned


def _ramp(n=100, freq="1h"):
    idx = pd.date_range("2025-01-01", periods=n, freq=freq)
    return pd.DataFrame({"soil_moisture": np.arange(n, dtype=float)}, index=idx)


def test_target_is_value_24_hours_later():
    df = _ramp(100)
    target = add_target(df, "soil_moisture", pd.Timedelta("1h"))
    assert target.iloc[0] == 24.0
    assert target.iloc[10] == 34.0
    assert target.iloc[-24:].isna().all()  # no future observation exists


def test_target_uses_timestamps_not_row_positions_across_gaps():
    df = _ramp(100).drop(_ramp(100).index[30:40])  # 10 h gap
    target = add_target(df, "soil_moisture", pd.Timedelta("1h"))
    # a row whose t+24h falls inside the gap must have NaN, not the value 24 rows later
    t = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=10)  # target time = hour 34, inside gap
    assert np.isnan(target.loc[t])
    t_ok = pd.Timestamp("2025-01-01") + pd.Timedelta(hours=50)
    assert target.loc[t_ok] == 74.0


def test_shifted_values_requires_sorted_unique_index():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-01 01:00", "2025-01-01 00:00", "2025-01-01 02:00"]))
    with pytest.raises(ValueError):
        shifted_values(s, pd.Timedelta("1h"), pd.Timedelta("30min"))


def test_lag_features_look_backwards():
    df, mapping, interval = synthetic_cleaned(days=5, freq="1h")
    feats = build_features(df, mapping, interval, include_target=False)
    row = feats.index[40]
    assert feats.loc[row, "sm_lag_3h"] == df["soil_moisture"].loc[row - pd.Timedelta("3h")]
    expected_change = feats.loc[row, "sm_current"] - feats.loc[row, "sm_lag_3h"]
    assert feats.loc[row, "sm_change_3h"] == expected_change


def test_features_do_not_depend_on_future_observations():
    """Changing data after time t must not change the features of the row at t."""
    df, mapping, interval = synthetic_cleaned(days=10, freq="1h")
    cut = df.index[150]
    full = build_features(df, mapping, interval, include_target=False)
    altered = df.copy()
    altered.loc[altered.index > cut] = altered.loc[altered.index > cut] * 3 + 7
    changed = build_features(altered, mapping, interval, include_target=False)
    pd.testing.assert_series_equal(full.loc[cut], changed.loc[cut])


def test_target_column_is_not_among_features():
    df, mapping, interval = synthetic_cleaned(days=10, freq="1h")
    frame = build_features(df, mapping, interval, include_target=True)
    assert config.TARGET_COLUMN in frame.columns
    assert config.TARGET_COLUMN not in feature_columns(frame)
    assert frame[config.TARGET_COLUMN].notna().all()


def test_unmapped_optional_roles_are_skipped():
    df, mapping, interval = synthetic_cleaned(days=6, freq="1h")
    mapping = {**mapping, "humidity": None, "irrigation": None}
    frame = build_features(df, mapping, interval, include_target=False)
    assert "hum_current" not in frame.columns
    assert "irrigation_sum_24h" not in frame.columns
    assert "temp_current" in frame.columns


def test_missing_soil_moisture_column_raises():
    df, mapping, interval = synthetic_cleaned(days=5, freq="1h")
    with pytest.raises(KeyError):
        build_features(df.drop(columns=["soil_moisture"]), mapping, interval)


def test_chronological_split_is_ordered_non_overlapping_and_purged():
    df, mapping, interval = synthetic_cleaned(days=40, freq="1h")
    frame = build_features(df, mapping, interval)
    parts = chronological_split(frame)
    horizon = pd.Timedelta(hours=config.FORECAST_HORIZON_HOURS)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    assert tr.index.max() < va.index.min() < va.index.max() < te.index.min()
    # targets of the last training rows must be realised before validation begins
    assert tr.index.max() + horizon <= va.index.min()
    assert va.index.max() + horizon <= te.index.min()
    assert len(tr) > len(va) and len(tr) > len(te)
    assert (tr.index.is_monotonic_increasing and va.index.is_monotonic_increasing and te.index.is_monotonic_increasing)


def test_chronological_split_rejects_unsorted_and_tiny_inputs():
    df, mapping, interval = synthetic_cleaned(days=40, freq="1h")
    frame = build_features(df, mapping, interval)
    with pytest.raises(ValueError):
        chronological_split(frame.iloc[::-1])
    with pytest.raises(ValueError):
        chronological_split(frame.iloc[:10])
    with pytest.raises(ValueError):
        chronological_split(frame, train_frac=0.9, val_frac=0.2)
