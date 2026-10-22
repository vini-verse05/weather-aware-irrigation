"""End-to-end training/evaluation checks on SYNTHETIC data (software tests only).

The numbers produced here are meaningless and must never be quoted as results.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config, train_model
from src.evaluate_model import comparison_table, persistence_predict, regenerate_figures, regression_metrics
from tests.helpers import trained_model_dir, write_synthetic_csv
import tests.helpers as helpers


def test_regression_metrics_match_hand_computation():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert abs(m["MAE"] - 2 / 3) < 1e-12
    assert abs(m["RMSE"] - (4 / 3) ** 0.5) < 1e-12
    assert abs(m["R2"] - (1 - 4 / 2)) < 1e-12


def test_metrics_reject_bad_shapes_and_handle_constant_targets():
    with pytest.raises(ValueError):
        regression_metrics([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        regression_metrics([], [])
    assert np.isnan(regression_metrics([2, 2, 2], [2, 2, 3])["R2"])


def test_persistence_baseline_returns_current_value():
    X = pd.DataFrame({"sm_current": [1.0, 2.0, 3.0], "other": [9, 9, 9]})
    assert persistence_predict(X).tolist() == [1.0, 2.0, 3.0]


def test_comparison_table_flattens_results():
    t = comparison_table({"A": {"val": {"MAE": 1, "RMSE": 2, "R2": 0.5}, "test": {"MAE": 3, "RMSE": 4, "R2": 0.1}}})
    assert t.loc[0, "val_RMSE"] == 2 and t.loc[0, "test_MAE"] == 3 and t.loc[0, "model"] == "A"


def test_training_run_writes_artifacts_with_real_metrics():
    models_dir = trained_model_dir()
    reports_dir = helpers._TRAINED_DIR_CACHE["reports"]
    import os
    for name in (config.MODEL_FILENAME, config.METADATA_FILENAME):
        assert os.path.exists(os.path.join(models_dir, name))
    for name in (config.COMPARISON_FILENAME, config.TEST_PREDICTIONS_FILENAME, config.RESULTS_SUMMARY_FILENAME):
        assert os.path.exists(os.path.join(reports_dir, name))
    table = pd.read_csv(os.path.join(reports_dir, config.COMPARISON_FILENAME))
    assert set(table["model"]) == {"Persistence baseline", "Linear Regression", "Random Forest", "Gradient Boosting"}
    meta = json.load(open(os.path.join(models_dir, config.METADATA_FILENAME)))
    # table and metadata must agree (numbers come from the same run)
    rf = table.set_index("model").loc["Random Forest", "test_RMSE"]
    assert abs(rf - meta["metrics"]["Random Forest"]["test"]["RMSE"]) < 1e-9
    preds = pd.read_csv(os.path.join(reports_dir, config.TEST_PREDICTIONS_FILENAME), parse_dates=["timestamp"])
    assert preds["timestamp"].is_monotonic_increasing
    assert preds[["actual", "predicted", "persistence"]].notna().all().all()


def test_training_is_reproducible(tmp_path):
    csv = write_synthetic_csv(tmp_path / "s.csv", days=25, freq="1h", seed=3)
    a = train_model.run(csv, tmp_path / "m1", tmp_path / "r1", make_plots=False, overrides={}, verbose=False)
    b = train_model.run(csv, tmp_path / "m2", tmp_path / "r2", make_plots=False, overrides={}, verbose=False)
    # Same seed -> same results up to floating-point rounding (parallel trees sum in varying order).
    for model, splits in a["metrics"].items():
        for split, metrics in splits.items():
            for name, value in metrics.items():
                assert abs(value - b["metrics"][model][split][name]) < 1e-9, (model, split, name)
    assert a["best_model"] == b["best_model"]


def test_regenerate_figures_from_saved_results(tmp_path):
    trained_model_dir()
    reports = helpers._TRAINED_DIR_CACHE["reports"]
    written = regenerate_figures(trained_model_dir(), reports)
    assert len(written) == 4 and all(p.exists() for p in written)


def test_regenerate_figures_without_results_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model"):
        regenerate_figures(tmp_path, tmp_path)


def test_training_without_dataset_reports_pending_and_trains_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "empty")
    monkeypatch.setattr(config, "LOCAL_CONFIG_PATH", tmp_path / "no_local_config.json")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    code = train_model.main([])
    assert code == 2
    assert not (tmp_path / "models" / config.MODEL_FILENAME).exists()


def test_too_little_data_is_rejected(tmp_path):
    csv = write_synthetic_csv(tmp_path / "tiny.csv", days=3, freq="1h")
    with pytest.raises(Exception):
        train_model.run(csv, tmp_path / "m", tmp_path / "r", make_plots=False, overrides={}, verbose=False)
