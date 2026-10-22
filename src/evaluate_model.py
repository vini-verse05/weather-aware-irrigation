"""Metrics, comparison tables and plots for the trained models.

Plot functions return matplotlib Figures (used both by the training script,
which saves PNG files, and by the Streamlit dashboard).

``python -m src.evaluate_model`` regenerates the figures from the saved test
predictions and metadata (no retraining).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; Streamlit renders figures itself
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from . import config  # noqa: E402


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R2. Returns NaN R2 when it is undefined (e.g. constant y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same shape.")
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 and np.ptp(y_true) > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def persistence_predict(X: pd.DataFrame) -> np.ndarray:
    """Baseline: the latest observed soil moisture is the 24 h forecast."""
    return X["sm_current"].to_numpy(dtype=float)


def comparison_table(results: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Flatten ``{model: {split: metrics}}`` into a tidy table."""
    rows = []
    for model, splits in results.items():
        row = {"model": model}
        for split, m in splits.items():
            for k, v in m.items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_actual_vs_predicted_time(times, y_true, y_pred, title: str = "Actual vs predicted (test set)", unit: str = ""):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, y_true, label="Actual", linewidth=1.2)
    ax.plot(times, y_pred, label="Predicted", linewidth=1.0, alpha=0.85)
    ax.set_xlabel("Prediction timestamp")
    ax.set_ylabel(f"Soil moisture after 24 h {unit}".strip())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted_scatter(y_true, y_pred, title: str = "Actual vs predicted (test set)", unit: str = ""):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel(f"Actual {unit}".strip())
    ax.set_ylabel(f"Predicted {unit}".strip())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(times, y_true, y_pred, title: str = "Residuals (actual - predicted)"):
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(resid, bins=40)
    axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("Residual distribution")
    axes[0].set_xlabel("Residual")
    axes[1].scatter(y_pred, resid, s=6, alpha=0.4)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Residual vs predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[2].plot(times, resid, linewidth=0.8)
    axes[2].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[2].set_title("Residual over time")
    axes[2].tick_params(axis="x", rotation=30)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame, top: int = 15, title: str = "Feature importance"):
    """``importance`` needs columns ``feature`` and ``importance`` (optional ``std``)."""
    imp = importance.sort_values("importance", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(imp) + 1)))
    err = imp["std"] if "std" in imp.columns else None
    ax.barh(imp["feature"], imp["importance"], xerr=err)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_model_comparison(table: pd.DataFrame, metric: str = "test_RMSE"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(table["model"], table[metric])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Model comparison ({metric.replace('_', ' ')}, lower is better)" if "R2" not in metric else f"Model comparison ({metric.replace('_', ' ')})")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI: regenerate figures from saved results
# --------------------------------------------------------------------------
def regenerate_figures(models_dir: Path | None = None, reports_dir: Path | None = None) -> list[Path]:
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    reports_dir = Path(reports_dir) if reports_dir else config.REPORTS_DIR
    meta_path = models_dir / config.METADATA_FILENAME
    pred_path = reports_dir / config.TEST_PREDICTIONS_FILENAME
    if not meta_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            "Saved results not found. Run `python -m src.train_model` first "
            f"(looked for {meta_path.name} and {pred_path.name})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    preds = pd.read_csv(pred_path, parse_dates=["timestamp"])
    unit = f"({meta.get('soil_moisture_unit', '')})" if meta.get("soil_moisture_unit") else ""
    fig_dir = reports_dir / "figures"
    best = meta["best_model"]
    written: list[Path] = []
    for fn, fig in (
        ("actual_vs_predicted_time.png", plot_actual_vs_predicted_time(preds["timestamp"], preds["actual"], preds["predicted"], f"{best}: actual vs predicted (test)", unit)),
        ("actual_vs_predicted_scatter.png", plot_actual_vs_predicted_scatter(preds["actual"], preds["predicted"], f"{best}: actual vs predicted (test)", unit)),
        ("residuals.png", plot_residuals(preds["timestamp"], preds["actual"], preds["predicted"])),
    ):
        save_figure(fig, fig_dir / fn)
        written.append(fig_dir / fn)
    if meta.get("feature_importance"):
        imp = pd.DataFrame(meta["feature_importance"])
        save_figure(plot_feature_importance(imp, title=f"{best}: permutation importance (validation)"), fig_dir / "feature_importance.png")
        written.append(fig_dir / "feature_importance.png")
    return written


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Regenerate evaluation figures from saved results.").parse_args(argv)
    try:
        for p in regenerate_figures():
            print(f"wrote {p}")
    except FileNotFoundError as exc:
        print(exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
