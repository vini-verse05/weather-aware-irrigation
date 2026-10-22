"""Prediction output validation, input validation and model loading.

A small model is trained on SYNTHETIC data purely to exercise the code paths.
"""

import json
import math
import shutil

import pandas as pd
import pytest

from src import config
from src.predict import ModelNotFoundError, describe_change, load_model, predict_from_history, predict_soil_moisture
from src.preprocessing import InvalidInputError, make_pipeline, validate_feature_values
from tests.helpers import make_synthetic_frame, trained_model_dir


def _bundle():
    return load_model(trained_model_dir())


def _medians(bundle):
    return {k: v["median"] for k, v in bundle.metadata["feature_stats"].items()}


def test_prediction_output_is_valid():
    bundle = _bundle()
    result = predict_soil_moisture(bundle, _medians(bundle))
    assert isinstance(result.predicted, float) and math.isfinite(result.predicted)
    assert result.horizon_hours == config.FORECAST_HORIZON_HOURS
    assert result.direction in {"increase", "decrease", "no clear change"}
    assert abs(result.change - (result.predicted - result.current)) < 1e-9
    assert result.model_name == bundle.name


def test_missing_required_input_is_rejected():
    bundle = _bundle()
    values = _medians(bundle)
    del values["sm_current"]
    with pytest.raises(InvalidInputError, match="sm_current"):
        predict_soil_moisture(bundle, values)


@pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), -float("inf"), None, True, [1, 2]])
def test_invalid_required_values_are_rejected(bad):
    bundle = _bundle()
    values = {**_medians(bundle), "sm_current": bad}
    with pytest.raises(InvalidInputError):
        predict_soil_moisture(bundle, values)


def test_invalid_optional_value_is_rejected_but_missing_optional_is_imputed_with_warning():
    bundle = _bundle()
    values = _medians(bundle)
    with pytest.raises(InvalidInputError):
        predict_soil_moisture(bundle, {**values, "temp_current": "hot"})
    optional = next(f for f in bundle.feature_names if f.startswith("temp_"))
    values.pop(optional)
    result = predict_soil_moisture(bundle, values)
    assert math.isfinite(result.predicted)
    assert any(optional in w for w in result.warnings)


def test_out_of_range_input_gives_warning_not_error():
    bundle = _bundle()
    values = {**_medians(bundle), "sm_current": 1e6}
    result = predict_soil_moisture(bundle, values)
    assert any("training range" in w for w in result.warnings)


def test_non_mapping_input_is_rejected():
    bundle = _bundle()
    with pytest.raises(InvalidInputError):
        predict_soil_moisture(bundle, [1, 2, 3])


def test_unknown_inputs_are_ignored_with_warning():
    bundle = _bundle()
    result = predict_soil_moisture(bundle, {**_medians(bundle), "not_a_feature": 1.0})
    assert any("unknown" in w.lower() for w in result.warnings)


def test_validate_feature_values_returns_one_ordered_row():
    row, warnings = validate_feature_values({"a": 1, "b": "2"}, ["b", "a"], required=("a",))
    assert list(row.columns) == ["b", "a"]
    assert row.iloc[0].tolist() == [2.0, 1.0]
    assert warnings == []


def test_describe_change():
    assert describe_change(1.0, 0.5) == "increase"
    assert describe_change(-1.0, 0.5) == "decrease"
    assert describe_change(0.2, 0.5) == "no clear change"
    assert describe_change(None, 0.5) == "unknown"
    assert describe_change(float("nan"), 0.5) == "unknown"


def test_loading_missing_model_raises_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="train_model"):
        load_model(tmp_path)


def test_loading_corrupted_model_raises(tmp_path):
    src = trained_model_dir()
    shutil.copy(f"{src}/{config.METADATA_FILENAME}", tmp_path / config.METADATA_FILENAME)
    (tmp_path / config.MODEL_FILENAME).write_bytes(b"this is not a model")
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path)


def test_incomplete_metadata_raises(tmp_path):
    src = trained_model_dir()
    shutil.copy(f"{src}/{config.MODEL_FILENAME}", tmp_path / config.MODEL_FILENAME)
    (tmp_path / config.METADATA_FILENAME).write_text(json.dumps({"best_model": "x"}))
    with pytest.raises(ModelNotFoundError, match="feature_names"):
        load_model(tmp_path)


def test_model_metadata_records_real_run_details():
    meta = _bundle().metadata
    assert meta["forecast_horizon_hours"] == 24
    assert set(meta["metrics"]) >= {"Persistence baseline", "Linear Regression", "Random Forest", "Gradient Boosting"}
    assert meta["best_model"] in {"Linear Regression", "Random Forest", "Gradient Boosting"}
    # model selection must follow validation RMSE
    ml = {k: v["val"]["RMSE"] for k, v in meta["metrics"].items() if k != "Persistence baseline"}
    assert meta["best_model"] == min(ml, key=ml.get)
    # splits are chronological
    s = meta["split"]
    assert s["train"]["end"] < s["val"]["start"] < s["val"]["end"] < s["test"]["start"]


def test_predict_from_history_works_and_requires_enough_data():
    bundle = _bundle()
    history = make_synthetic_frame(days=4, freq="1h", seed=5)
    result = predict_from_history(bundle, history)
    assert math.isfinite(result.predicted)
    with pytest.raises(InvalidInputError):
        predict_from_history(bundle, history.head(10))  # < 24 h of history
    with pytest.raises(InvalidInputError):
        predict_from_history(bundle, history.drop(columns=["soil_moisture"]))


def test_pipeline_imputer_is_fitted_on_training_data_only():
    from sklearn.linear_model import LinearRegression

    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
    pipe = make_pipeline(LinearRegression())
    pipe.fit(train, [1.0, 2.0, 3.0, 2.0])
    stats = pipe.named_steps["imputer"].statistics_
    assert stats[0] == 2.0  # median of the TRAINING column only
    later = pd.DataFrame({"x": [1000.0, None]})
    pipe.predict(later)
    assert pipe.named_steps["imputer"].statistics_[0] == 2.0  # unchanged by later data
