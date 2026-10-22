"""Preprocessing pipelines and input validation.

Imputation and scaling live INSIDE the scikit-learn Pipeline, so they are fitted
on the training split only and re-used unchanged for validation, test and
live predictions (no leakage from later data).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class InvalidInputError(ValueError):
    """Raised when feature values supplied for prediction are unusable."""


def make_pipeline(model: RegressorMixin, scale: bool = False) -> Pipeline:
    """Median imputation (+ optional standardisation) followed by ``model``."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def compute_feature_stats(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summary statistics of the TRAINING features (for form defaults and range checks)."""
    stats: dict[str, dict[str, float]] = {}
    for col in X.columns:
        s = X[col].dropna()
        if s.empty:
            continue
        stats[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
        }
    return stats


def validate_feature_values(
    values: Mapping[str, Any],
    feature_names: list[str],
    required: tuple[str, ...] = ("sm_current",),
    stats: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate a dict of feature values and return a one-row frame plus warnings.

    * Required features must be present and finite numbers -> else InvalidInputError.
    * Optional features that are missing/None/blank become NaN (the pipeline's
      imputer fills them with a training median) and generate a warning.
    * Non-numeric or non-finite values raise InvalidInputError.
    * Values outside the training range do not fail but generate a warning,
      because predictions there are extrapolations.
    """
    if not isinstance(values, Mapping):
        raise InvalidInputError("Feature values must be provided as a mapping of name -> number.")
    warnings: list[str] = []
    row: dict[str, float] = {}
    not_provided: list[str] = []
    out_of_range: list[str] = []
    unknown = [k for k in values if k not in feature_names]
    if unknown:
        warnings.append(f"Ignored unknown inputs: {unknown}")

    for name in feature_names:
        raw = values.get(name, None)
        blank = raw is None or (isinstance(raw, str) and raw.strip() == "") or (
            isinstance(raw, float) and math.isnan(raw)
        )
        if blank:
            if name in required:
                raise InvalidInputError(f"Required input '{name}' is missing.")
            row[name] = np.nan
            not_provided.append(name)
            continue
        if isinstance(raw, bool):
            raise InvalidInputError(f"Input '{name}' must be a number, got a boolean.")
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(f"Input '{name}' must be numeric, got {raw!r}.") from exc
        if not math.isfinite(val):
            raise InvalidInputError(f"Input '{name}' must be a finite number, got {raw!r}.")
        row[name] = val
        if stats and name in stats:
            lo, hi = stats[name]["min"], stats[name]["max"]
            if val < lo or val > hi:
                out_of_range.append(f"{name}={val:g} (training range {lo:g} to {hi:g})")
    if not_provided:
        warnings.append(f"Not provided, training-set median used instead: {not_provided}")
    if out_of_range:
        warnings.append(
            "Outside the training range (the prediction is an extrapolation and may be unreliable): "
            + "; ".join(out_of_range)
        )
    return pd.DataFrame([row], columns=feature_names), warnings
