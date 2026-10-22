"""Reproducible training pipeline.

    python -m src.train_model [--data-file PATH]

Steps: load + clean data -> build features/target -> chronological split
(with purge gap) -> fit persistence, Linear Regression, Random Forest and
Gradient Boosting -> choose the best ML model by VALIDATION RMSE -> report
held-out TEST metrics -> save model, metadata, tables and plots.

Nothing is hard-coded: every reported number comes from this run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from . import config
from .data_loader import DataValidationError, load_dataset
from .evaluate_model import (
    comparison_table,
    persistence_predict,
    plot_actual_vs_predicted_scatter,
    plot_actual_vs_predicted_time,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    regression_metrics,
    save_figure,
)
from .feature_engineering import build_features, chronological_split, feature_columns
from .preprocessing import compute_feature_stats, make_pipeline

PERSISTENCE = "Persistence baseline"


def candidate_models() -> dict[str, Any]:
    """Scikit-learn pipelines to compare (all fitted on the training split only)."""
    rs = config.RANDOM_STATE
    return {
        "Linear Regression": make_pipeline(LinearRegression(), scale=True),
        "Random Forest": make_pipeline(
            RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.6, n_jobs=-1, random_state=rs)
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=rs)
        ),
    }


def _model_importance(pipe, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    """Permutation importance on the validation set (works for every model type)."""
    result = permutation_importance(
        pipe, X_val, y_val, n_repeats=5, random_state=config.RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": X_val.columns, "importance": result.importances_mean, "std": result.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run(
    data_file: str | Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    make_plots: bool = True,
    overrides: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline and return the metadata dictionary that was saved."""
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    reports_dir = Path(reports_dir) if reports_dir else config.REPORTS_DIR
    log = print if verbose else (lambda *a, **k: None)

    if overrides is None:
        overrides = config.load_local_overrides()
    data = load_dataset(data_file, overrides)
    mapping = data.mapping
    log(f"Dataset file     : {data.path.name}")
    log(f"Sampling interval: {data.interval}  |  rows after cleaning: {len(data.df)}")
    log("Column mapping   : " + ", ".join(f"{k}={v}" for k, v in mapping.items() if v))
    log("(Verify this mapping against the dataset's data dictionary; override in config.local.json.)")

    frame = build_features(data.df, mapping, data.interval, include_target=True)
    if len(frame) < 100:
        raise DataValidationError(f"Only {len(frame)} usable samples after feature/target construction; need >= 100.")
    splits = chronological_split(frame)
    feats = feature_columns(splits["train"])
    X = {k: v[feats] for k, v in splits.items()}
    y = {k: v[config.TARGET_COLUMN] for k, v in splits.items()}
    for k in splits:
        log(f"  {k:<5}: {len(splits[k]):>6} rows  {splits[k].index.min()} -> {splits[k].index.max()}")

    results: dict[str, dict[str, dict[str, float]]] = {
        PERSISTENCE: {s: regression_metrics(y[s], persistence_predict(X[s])) for s in ("val", "test")}
    }
    fitted: dict[str, Any] = {}
    for name, pipe in candidate_models().items():
        log(f"Training {name} ...")
        pipe.fit(X["train"], y["train"])
        fitted[name] = pipe
        results[name] = {s: regression_metrics(y[s], pipe.predict(X[s])) for s in ("val", "test")}

    best = min(fitted, key=lambda n: results[n]["val"]["RMSE"])
    best_pipe = fitted[best]
    beats_baseline = results[best]["val"]["RMSE"] < results[PERSISTENCE]["val"]["RMSE"]
    table = comparison_table(results)
    log("\n" + table.round(4).to_string(index=False))
    log(f"\nSelected model (lowest validation RMSE): {best}")
    if not beats_baseline:
        log("NOTE: the selected model does not beat the persistence baseline on validation data.")

    importance = _model_importance(best_pipe, X["val"], y["val"])
    test_pred = best_pipe.predict(X["test"])
    sm_train = splits["train"]["sm_current"]

    metadata: dict[str, Any] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_file": data.path.name,
        "sector": data.report.get("sector"),
        "column_mapping": mapping,
        "sampling_interval": str(data.interval),
        "forecast_horizon_hours": config.FORECAST_HORIZON_HOURS,
        "soil_moisture_unit": overrides.get("soil_moisture_unit") or config.DEFAULT_SOIL_MOISTURE_UNIT,
        "feature_names": feats,
        "feature_stats": compute_feature_stats(X["train"]),
        "best_model": best,
        "best_beats_persistence_on_validation": bool(beats_baseline),
        "metrics": results,
        "split": {
            k: {"rows": int(len(v)), "start": str(v.index.min()), "end": str(v.index.max())}
            for k, v in splits.items()
        },
        "feature_importance": importance.head(20).to_dict(orient="records"),
        "suggested_thresholds": {
            "note": "Illustrative percentiles of TRAINING soil-moisture readings; NOT agronomic thresholds.",
            "critical_p05": float(sm_train.quantile(0.05)),
            "dry_p20": float(sm_train.quantile(0.20)),
        },
        "cleaning_report": {k: v for k, v in data.report.items() if k != "all_columns"},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, models_dir / config.MODEL_FILENAME)
    (models_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    table.to_csv(reports_dir / config.COMPARISON_FILENAME, index=False)
    pd.DataFrame({"timestamp": splits["test"].index, "actual": y["test"].to_numpy(), "predicted": test_pred,
                  "persistence": persistence_predict(X["test"])}).to_csv(reports_dir / config.TEST_PREDICTIONS_FILENAME, index=False)
    _write_results_summary(reports_dir / config.RESULTS_SUMMARY_FILENAME, metadata, table)

    if make_plots:
        unit = f"({metadata['soil_moisture_unit']})"
        fig_dir = reports_dir / "figures"
        t = splits["test"].index
        save_figure(plot_actual_vs_predicted_time(t, y["test"], test_pred, f"{best}: actual vs predicted (test)", unit), fig_dir / "actual_vs_predicted_time.png")
        save_figure(plot_actual_vs_predicted_scatter(y["test"], test_pred, f"{best}: actual vs predicted (test)", unit), fig_dir / "actual_vs_predicted_scatter.png")
        save_figure(plot_residuals(t, y["test"], test_pred), fig_dir / "residuals.png")
        save_figure(plot_feature_importance(importance, title=f"{best}: permutation importance (validation)"), fig_dir / "feature_importance.png")
        save_figure(plot_model_comparison(table, "test_RMSE"), fig_dir / "model_comparison_test_rmse.png")

    log(f"\nSaved model to {models_dir / config.MODEL_FILENAME}")
    log(f"Saved results to {reports_dir}")
    return metadata


def _write_results_summary(path: Path, meta: dict[str, Any], table: pd.DataFrame) -> None:
    """Auto-generated markdown with this run's actual numbers (safe to paste into the README)."""
    lines = [
        "# Results of the latest training run (auto-generated)",
        "",
        f"- Trained at (UTC): {meta['trained_at_utc']}",
        f"- Dataset file: `{meta['dataset_file']}`" + (f" (sector: {meta['sector']})" if meta.get("sector") else ""),
        f"- Forecast horizon: {meta['forecast_horizon_hours']} h; sampling interval: {meta['sampling_interval']}",
        f"- Soil-moisture unit: {meta['soil_moisture_unit']}",
        f"- Selected model (lowest validation RMSE): **{meta['best_model']}**",
        "- Splits (chronological, with a purge gap of one horizon): "
        + "; ".join(f"{k}: {v['rows']} rows ({v['start'][:10]} to {v['end'][:10]})" for k, v in meta["split"].items()),
        "",
        table.round(4).to_markdown(index=False) if _has_tabulate() else "```\n" + table.round(4).to_string(index=False) + "\n```",
        "",
        "Lower MAE/RMSE is better; R2 closer to 1 is better. Test metrics are from held-out, later-in-time data.",
    ]
    if not meta["best_beats_persistence_on_validation"]:
        lines.append("\n> The selected model did NOT beat the persistence baseline on validation data.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the 24 h soil-moisture models.")
    parser.add_argument("--data-file", help="CSV file to use (default: auto-detect under data/)")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    args = parser.parse_args(argv)
    try:
        run(data_file=args.data_file, make_plots=not args.no_plots)
    except DataValidationError as exc:
        print("\nTRAINING PENDING - dataset not ready.")
        print(f"Reason: {exc}")
        print("No model was trained and no results were produced.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
