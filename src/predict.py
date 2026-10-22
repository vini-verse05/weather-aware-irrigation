"""Load the trained model and produce validated 24-hour soil-moisture predictions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from . import config
from .data_loader import clean_timeseries
from .feature_engineering import build_features
from .preprocessing import InvalidInputError, validate_feature_values


class ModelNotFoundError(FileNotFoundError):
    """Raised when trained model artifacts are missing or unreadable."""


@dataclass
class ModelBundle:
    pipeline: Any
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])

    @property
    def name(self) -> str:
        return str(self.metadata.get("best_model", "model"))

    @property
    def unit(self) -> str:
        return str(self.metadata.get("soil_moisture_unit", ""))

    @property
    def validation_mae(self) -> float | None:
        try:
            return float(self.metadata["metrics"][self.name]["val"]["MAE"])
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class PredictionResult:
    predicted: float
    current: float | None
    change: float | None
    direction: str  # "increase" | "decrease" | "no clear change" | "unknown"
    model_name: str
    unit: str
    horizon_hours: int
    warnings: list[str] = field(default_factory=list)


def load_model(models_dir: Path | None = None) -> ModelBundle:
    """Load the saved pipeline and its metadata, with a helpful error if absent."""
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    model_path = models_dir / config.MODEL_FILENAME
    meta_path = models_dir / config.METADATA_FILENAME
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotFoundError(
            "No trained model found. Place the dataset in data/ and run `python -m src.train_model` "
            f"(expected {model_path.name} and {meta_path.name} in {models_dir})."
        )
    try:
        pipeline = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ModelNotFoundError(f"Model artifacts could not be read ({exc}). Re-run training.") from exc
    if "feature_names" not in metadata:
        raise ModelNotFoundError("Model metadata is incomplete (no feature_names). Re-run training.")
    return ModelBundle(pipeline=pipeline, metadata=metadata)


def describe_change(change: float | None, tolerance: float | None) -> str:
    """Classify a predicted change. Changes smaller than the model's validation MAE are
    within typical model error and reported as 'no clear change'."""
    if change is None or not math.isfinite(change):
        return "unknown"
    tol = tolerance if tolerance and tolerance > 0 else 0.0
    if abs(change) <= tol:
        return "no clear change"
    return "increase" if change > 0 else "decrease"


def predict_soil_moisture(bundle: ModelBundle, values: Mapping[str, Any]) -> PredictionResult:
    """Predict soil moisture ``horizon`` hours ahead from a dict of feature values.

    Raises InvalidInputError for unusable inputs. Out-of-range inputs are accepted
    with a warning.
    """
    row, warnings = validate_feature_values(
        values, bundle.feature_names, required=("sm_current",), stats=bundle.metadata.get("feature_stats")
    )
    pred = float(bundle.pipeline.predict(row)[0])
    if not math.isfinite(pred):
        raise InvalidInputError("The model returned a non-finite value for these inputs.")
    current = float(row["sm_current"].iloc[0]) if "sm_current" in row else None
    change = pred - current if current is not None else None
    return PredictionResult(
        predicted=pred,
        current=current,
        change=change,
        direction=describe_change(change, bundle.validation_mae),
        model_name=bundle.name,
        unit=bundle.unit,
        horizon_hours=int(bundle.metadata.get("forecast_horizon_hours", config.FORECAST_HORIZON_HOURS)),
        warnings=warnings,
    )


def predict_from_history(bundle: ModelBundle, history: pd.DataFrame) -> PredictionResult:
    """Predict from a raw history table (e.g. exported from an IoT logger).

    ``history`` must contain a timestamp column and the soil-moisture column (named as
    in the training data) and at least 24 h of past readings. Optional weather columns
    are used when present. This is the hook for integrating a separate IoT project.
    """
    mapping = bundle.metadata["column_mapping"]
    needed = [mapping.get("timestamp"), mapping.get("soil_moisture")]
    missing = [c for c in needed if c not in history.columns]
    if missing:
        raise InvalidInputError(f"History is missing required column(s): {missing}")
    present_mapping = {r: (c if c in history.columns else None) for r, c in mapping.items()}
    cleaned, _ = clean_timeseries(history, present_mapping)
    interval = pd.Timedelta(bundle.metadata["sampling_interval"])
    feats = build_features(cleaned, present_mapping, interval, include_target=False)
    if feats.empty:
        raise InvalidInputError("Not enough history (need at least 24 h of past soil-moisture readings).")
    last = feats.iloc[-1]
    result = predict_soil_moisture(bundle, {k: last.get(k) for k in bundle.feature_names})
    result.warnings.append(f"Features built from history ending at {feats.index[-1]}.")
    return result
