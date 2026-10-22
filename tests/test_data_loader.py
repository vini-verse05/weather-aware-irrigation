"""Dataset discovery, column mapping, validation and cleaning (synthetic fixtures only)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (
    ColumnMappingError,
    DataValidationError,
    clean_timeseries,
    detect_columns,
    detect_target_like_columns,
    find_candidate_files,
    infer_sampling_interval,
    load_dataset,
    validate_existing_target,
)
from tests.helpers import make_synthetic_frame, write_synthetic_csv


def test_detect_columns_maps_all_roles():
    raw = make_synthetic_frame(days=3)
    mapping, _ = detect_columns(list(raw.columns))
    assert mapping["timestamp"] == "timestamp"
    assert mapping["soil_moisture"] == "soil_moisture"
    assert mapping["temperature"] == "air_temperature"
    assert mapping["humidity"] == "relative_humidity"
    assert mapping["precipitation"] == "precipitation_mm"
    assert mapping["et0"] == "et0_mm"
    assert mapping["irrigation"] == "irrigation_volume"


def test_engineered_and_target_columns_are_never_selected_as_inputs():
    cols = ["timestamp", "soil_moisture_target_24h", "soil_moisture_lag_1h", "soil_moisture_roll_mean_6h", "soil_moisture"]
    mapping, _ = detect_columns(cols)
    assert mapping["soil_moisture"] == "soil_moisture"
    cols = ["timestamp", "soil_moisture_target_24h", "soil_moisture_lag_1h"]
    mapping, _ = detect_columns(cols)
    assert mapping["soil_moisture"] is None


def test_target_like_columns_are_reported():
    found = detect_target_like_columns(["timestamp", "soil_moisture", "soil_moisture_target_24h"])
    assert found == ["soil_moisture_target_24h"]


def test_override_is_used_and_validated():
    cols = ["time", "sensor_a", "sensor_b"]
    mapping, _ = detect_columns(cols, {"soil_moisture": "sensor_b"})
    assert mapping["soil_moisture"] == "sensor_b"
    with pytest.raises(ColumnMappingError):
        detect_columns(cols, {"soil_moisture": "does_not_exist"})


def test_clean_timeseries_sorts_merges_duplicates_and_blanks_invalid_values():
    raw = pd.DataFrame({
        "timestamp": ["2025-01-01 02:00", "2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 01:00", "not a date"],
        "soil_moisture": [30.0, 28.0, 29.0, 31.0, 99.0],
        "relative_humidity": [50.0, 150.0, 60.0, 60.0, 60.0],
        "precipitation_mm": [0.0, -1.0, 0.0, 0.0, 0.0],
    })
    mapping, _ = detect_columns(list(raw.columns))
    df, report = clean_timeseries(raw, mapping)
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert report["invalid_timestamps_dropped"] == 1
    assert report["duplicate_timestamps_merged"] == 1
    assert df.loc[pd.Timestamp("2025-01-01 01:00"), "soil_moisture"] == 30.0  # mean of 29 and 31
    assert np.isnan(df.loc[pd.Timestamp("2025-01-01 00:00"), "relative_humidity"])  # 150 % is impossible
    assert np.isnan(df.loc[pd.Timestamp("2025-01-01 00:00"), "precipitation_mm"])  # negative rain is impossible


def test_negative_soil_moisture_is_blanked():
    raw = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=5, freq="h").astype(str),
                        "soil_moisture": [10.0, -5.0, 11.0, 12.0, 13.0]})
    mapping, _ = detect_columns(list(raw.columns))
    df, report = clean_timeseries(raw, mapping)
    assert df["soil_moisture"].isna().sum() == 1
    assert report["invalid_values_blanked"]["soil_moisture"] == 1


def test_non_numeric_values_become_missing():
    raw = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=4, freq="h").astype(str),
                        "soil_moisture": ["10", "bad", "12", "13"]})
    mapping, _ = detect_columns(list(raw.columns))
    df, _ = clean_timeseries(raw, mapping)
    assert df["soil_moisture"].isna().sum() == 1


def test_all_invalid_soil_moisture_raises():
    raw = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=4, freq="h").astype(str),
                        "soil_moisture": ["x", "y", "z", "w"]})
    mapping, _ = detect_columns(list(raw.columns))
    with pytest.raises(DataValidationError):
        clean_timeseries(raw, mapping)


def test_missing_required_columns_raise():
    raw = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ColumnMappingError):
        clean_timeseries(raw, {"timestamp": None, "soil_moisture": None})


def test_infer_sampling_interval():
    idx = pd.date_range("2025-01-01", periods=50, freq="10min")
    interval, share = infer_sampling_interval(idx)
    assert interval == pd.Timedelta("10min") and share == 1.0
    with pytest.raises(DataValidationError):
        infer_sampling_interval(idx[:2])


def test_load_dataset_from_file_reports_interval(tmp_path):
    path = write_synthetic_csv(tmp_path / "series.csv", days=5, freq="30min")
    data = load_dataset(path, overrides={})
    assert data.interval == pd.Timedelta("30min")
    assert data.mapping["soil_moisture"] == "soil_moisture"
    assert data.report["file"] == "series.csv"


def test_missing_dataset_gives_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(DataValidationError, match="data/README.md"):
        load_dataset(overrides={})


def test_explicit_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError):
        load_dataset(tmp_path / "nope.csv", overrides={})


def test_find_candidate_files_prefers_time_series_over_event_tables(tmp_path):
    write_synthetic_csv(tmp_path / "sector1_timeseries_10min.csv", days=3)
    write_synthetic_csv(tmp_path / "valve_events_raw.csv", days=3)
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n")
    found = find_candidate_files(tmp_path)
    names = [p.name for p in found]
    assert names[0] == "sector1_timeseries_10min.csv"
    assert "notes.csv" not in names


def test_sector_column_requires_explicit_choice(tmp_path):
    df = make_synthetic_frame(days=3)
    df["sector"] = ["A"] * (len(df) // 2) + ["B"] * (len(df) - len(df) // 2)
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="sector_value"):
        load_dataset(path, overrides={"sector_column": "sector"})
    data = load_dataset(path, overrides={"sector_column": "sector", "sector_value": "A"})
    assert len(data.df) <= len(df) // 2


def test_validate_existing_target_detects_agreement_and_disagreement():
    raw = make_synthetic_frame(days=6, freq="1h")
    mapping, _ = detect_columns(list(raw.columns))
    good = raw["soil_moisture"].shift(-24)
    raw["soil_moisture_target_24h"] = good
    ok = validate_existing_target(raw, mapping, "soil_moisture_target_24h")
    assert ok["agreement"] == 1.0
    raw["soil_moisture_target_24h"] = good + 100.0
    bad = validate_existing_target(raw, mapping, "soil_moisture_target_24h")
    assert bad["agreement"] == 0.0
