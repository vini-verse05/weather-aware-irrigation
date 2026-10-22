"""Streamlit dashboard: Weather-Aware Smart Irrigation Prediction.

Run with:  python -m streamlit run app.py

The dashboard uses the real trained model (models/), the real dataset (data/)
and the live Open-Meteo API. When one of these is unavailable it shows a clear
message instead of fake values.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.data_loader import DataValidationError, load_dataset  # noqa: E402
from src.evaluate_model import (  # noqa: E402
    comparison_table,
    plot_actual_vs_predicted_scatter,
    plot_actual_vs_predicted_time,
    plot_feature_importance,
    plot_residuals,
)
from src.feature_engineering import build_features  # noqa: E402
from src.irrigation_recommendation import Recommendation, Thresholds, recommend  # noqa: E402
from src.predict import ModelNotFoundError, load_model, predict_soil_moisture  # noqa: E402
from src.preprocessing import InvalidInputError  # noqa: E402
from src.weather_service import (  # noqa: E402
    WeatherServiceError,
    get_forecast,
    get_historical,
    local_now,
    next_hours,
    summarize_rainfall,
)

st.set_page_config(page_title="Weather-Aware Irrigation", page_icon="🌱", layout="wide")

PAGES = [
    "Overview",
    "Soil-Moisture Prediction",
    "Weather Forecast",
    "Irrigation Recommendation",
    "Dataset Analysis",
    "Model Evaluation",
    "Project Information",
]

DATASET_CITATION = (
    "Adamo, T., Colizzi, L., Dimauro, G., Guerriero, E., Lomonte, N. Soil Moisture, Irrigation Actuator and "
    "Weather Dataset from a Multi-Sector Precision-Irrigation. Mendeley Data, V1. "
    "doi:10.17632/c837v6p8ph.1 (licence: CC BY 4.0)."
)


# ----------------------------------------------------------------------------
# Cached loaders (failures are never cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model_cached(model_mtime: float, meta_mtime: float):
    return load_model()


def get_model():
    """Return (bundle, None) or (None, message)."""
    mp = config.MODELS_DIR / config.MODEL_FILENAME
    meta = config.MODELS_DIR / config.METADATA_FILENAME
    if not (mp.exists() and meta.exists()):
        return None, (
            "No trained model found. Download the dataset into `data/` (see data/README.md) and run "
            "`python -m src.train_model`. Model training is PENDING."
        )
    try:
        return _load_model_cached(mp.stat().st_mtime, meta.stat().st_mtime), None
    except ModelNotFoundError as exc:
        return None, str(exc)


@st.cache_data(show_spinner="Loading and cleaning the dataset...")
def _load_dataset_cached(signature: str):
    data = load_dataset()
    return data.df, data.mapping, data.report, str(data.interval)


def get_dataset():
    """Return ((df, mapping, report, interval), None) or (None, message)."""
    try:
        files = [p for p in config.DATA_DIR.rglob("*.csv")] if config.DATA_DIR.exists() else []
        signature = "|".join(f"{p}:{p.stat().st_mtime}" for p in sorted(files))
        local = config.LOCAL_CONFIG_PATH.stat().st_mtime if config.LOCAL_CONFIG_PATH.exists() else 0
        return _load_dataset_cached(signature + f"|{local}"), None
    except (DataValidationError, ValueError) as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _latest_features_cached(signature: str):
    data = load_dataset()
    feats = build_features(data.df, data.mapping, data.interval, include_target=False)
    if feats.empty:
        raise DataValidationError("Not enough data to build features.")
    return feats.iloc[-1].to_dict(), str(feats.index[-1])


def get_latest_features():
    try:
        files = [p for p in config.DATA_DIR.rglob("*.csv")] if config.DATA_DIR.exists() else []
        sig = "|".join(f"{p}:{p.stat().st_mtime}" for p in sorted(files))
        return _latest_features_cached(sig), None
    except (DataValidationError, ValueError) as exc:
        return None, str(exc)


def fetch_weather(lat: float, lon: float, refresh: bool = False):
    """Return (WeatherData, None) or (None, message). Never raises."""
    try:
        return get_forecast(lat, lon, forecast_days=3, use_cache=not refresh), None
    except WeatherServiceError as exc:
        return None, str(exc)


def load_metadata() -> dict | None:
    p = config.MODELS_DIR / config.METADATA_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def sidebar() -> str:
    st.sidebar.title("🌱 Smart Irrigation")
    page = st.sidebar.radio("Section", PAGES)
    with st.sidebar.expander("Location for weather forecast", expanded=False):
        st.text_input("Location name", value=config.DEFAULT_LOCATION["name"], key="loc_name")
        st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=float(config.DEFAULT_LOCATION["latitude"]),
                        format="%.4f", key="loc_lat")
        st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=float(config.DEFAULT_LOCATION["longitude"]),
                        format="%.4f", key="loc_lon")
        st.caption("Editable. The default is the dataset's experimental farm (approximate).")
    st.sidebar.caption("Decision support only. This app never controls pumps or valves.")
    return page


# ----------------------------------------------------------------------------
# A. Overview
# ----------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Weather-Aware Smart Irrigation Prediction")
    st.write(
        "A machine-learning application that predicts **soil moisture 24 hours ahead** from recent sensor and weather "
        "history, then combines that prediction with a live **weather forecast** to give an explained irrigation "
        "suggestion. It is software-only: no ESP32, pump or other hardware is required, and nothing is switched automatically."
    )
    st.subheader("Objectives")
    st.markdown(
        "- Predict soil moisture 24 h ahead and whether it will rise or fall.\n"
        "- Check the forecast for upcoming rainfall.\n"
        "- Suggest: monitor, irrigation may be needed, delay because of rain, or insufficient information.\n"
        "- Show which inputs influence the prediction."
    )
    st.subheader("Workflow")
    st.markdown(
        "1. **Data** - public precision-irrigation dataset, cleaned and sorted chronologically.\n"
        "2. **Features** - current/lagged soil moisture, changes, past temperature, humidity, rainfall, "
        "evapotranspiration, irrigation, and time-of-day/season (past data only).\n"
        "3. **Models** - persistence baseline, Linear Regression, Random Forest, Gradient Boosting; chronological split.\n"
        "4. **Weather** - Open-Meteo forecast for the chosen location (used by the recommendation rules, *not* a model input).\n"
        "5. **Recommendation** - transparent, configurable rules, separate from the ML model."
    )
    st.subheader("Dataset")
    st.write(DATASET_CITATION)
    st.subheader("Model status")
    bundle, err = get_model()
    if bundle is None:
        st.warning(err)
    else:
        m = bundle.metadata
        st.success(f"Trained model available: **{bundle.name}** (trained {m.get('trained_at_utc', 'unknown')}).")
        test = m["metrics"].get(bundle.name, {}).get("test", {})
        if test:
            c1, c2, c3 = st.columns(3)
            c1.metric("Test MAE", f"{test['MAE']:.4g}")
            c2.metric("Test RMSE", f"{test['RMSE']:.4g}")
            c3.metric("Test R²", f"{test['R2']:.3f}")
        st.caption("Metrics are measured on held-out, later-in-time data from the training run on this machine.")
    ds, derr = get_dataset()
    if ds is None:
        st.info(f"Dataset: {derr}")
    else:
        st.info(f"Dataset loaded: {ds[2]['file']} ({ds[2]['rows_out']} rows, {ds[2]['start'][:10]} to {ds[2]['end'][:10]}).")


# ----------------------------------------------------------------------------
# B. Prediction
# ----------------------------------------------------------------------------
GROUPS = [
    ("Soil-moisture history", ("sm_",)),
    ("Recent weather (observed in the past)", ("temp_", "hum_", "precip_", "et0_")),
    ("Irrigation history", ("irrigation_",)),
]


def page_prediction() -> None:
    st.header("Soil-Moisture Prediction (24 h ahead)")
    bundle, err = get_model()
    if bundle is None:
        st.warning(err)
        return
    meta = bundle.metadata
    stats = meta.get("feature_stats", {})
    feats = bundle.feature_names
    unit = bundle.unit

    with st.expander("Model information"):
        st.write(f"**Model:** {bundle.name}  |  **Horizon:** {meta.get('forecast_horizon_hours')} h  |  "
                 f"**Sampling interval:** {meta.get('sampling_interval')}")
        st.write(f"**Trained on:** {meta.get('dataset_file')}  |  **Unit:** {unit}")
        val = meta["metrics"].get(bundle.name, {}).get("val", {})
        if val:
            st.write(f"**Validation error:** MAE {val['MAE']:.4g}, RMSE {val['RMSE']:.4g}")
        st.caption("Inputs are PAST observations. Forecast weather is not a model input.")

    start_from = st.radio("Start from", ["Training-data median values", "Most recent row of the dataset"], horizontal=True)
    defaults = {k: v["median"] for k, v in stats.items()}
    when = datetime.now()
    if start_from.startswith("Most recent"):
        latest, lerr = get_latest_features()
        if latest is None:
            st.warning(f"Could not read the dataset ({lerr}); using training medians instead.")
        else:
            defaults.update({k: v for k, v in latest[0].items() if v == v})
            when = pd.Timestamp(latest[1]).to_pydatetime()
            st.caption(f"Using the dataset's latest usable timestamp: {latest[1]}")
    mode = "latest" if start_from.startswith("Most recent") else "median"

    time_features = {"hour_sin", "hour_cos", "doy_sin", "doy_cos"} & set(feats)
    values: dict[str, float] = {}
    with st.form("predict_form"):
        st.write("Enter the model inputs (units are those of the training dataset).")
        for title, prefixes in GROUPS:
            names = [f for f in feats if f.startswith(prefixes)]
            if not names:
                continue
            st.markdown(f"**{title}**")
            cols = st.columns(3)
            for i, name in enumerate(names):
                lo, hi = stats.get(name, {}).get("min"), stats.get(name, {}).get("max")
                help_txt = f"Training range: {lo:.4g} to {hi:.4g}" if lo is not None else None
                values[name] = cols[i % 3].number_input(
                    name, value=float(defaults.get(name, 0.0)), format="%.4f", key=f"in_{mode}_{name}", help=help_txt)
        if time_features:
            st.markdown("**Prediction time** (converted to cyclical time-of-day / season features)")
            c1, c2 = st.columns(2)
            d = c1.date_input("Date", value=when.date(), key=f"in_{mode}_date")
            t = c2.time_input("Time", value=dtime(when.hour, when.minute), key=f"in_{mode}_time")
            hours = t.hour + t.minute / 60.0
            doy = pd.Timestamp(d).dayofyear
            values.update({"hour_sin": float(np.sin(2 * np.pi * hours / 24)), "hour_cos": float(np.cos(2 * np.pi * hours / 24)),
                           "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)), "doy_cos": float(np.cos(2 * np.pi * doy / 365.25))})
        submitted = st.form_submit_button("Predict soil moisture")

    if submitted:
        try:
            result = predict_soil_moisture(bundle, values)
        except InvalidInputError as exc:
            st.error(f"Invalid input: {exc}")
            return
        st.session_state["prediction"] = result
        st.session_state["prediction_id"] = st.session_state.get("prediction_id", 0) + 1

    result = st.session_state.get("prediction")
    if result is not None:
        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Predicted soil moisture in {result.horizon_hours} h", f"{result.predicted:.4g}",
                  delta=f"{result.change:+.3g}" if result.change is not None else None)
        c2.metric("Current soil moisture", f"{result.current:.4g}" if result.current is not None else "n/a")
        c3.metric("Expected direction", result.direction)
        st.caption(f"Unit: {result.unit}. Model: {result.model_name}.")
        mae = bundle.validation_mae
        if mae is not None:
            st.caption(f"Changes smaller than the model's validation MAE ({mae:.3g}) are reported as 'no clear change'.")
        for w in result.warnings:
            st.warning(w)
        st.info("Go to *Irrigation Recommendation* to combine this prediction with the weather forecast.")


# ----------------------------------------------------------------------------
# C. Weather
# ----------------------------------------------------------------------------
def location() -> tuple[str, float, float]:
    return (st.session_state.get("loc_name", config.DEFAULT_LOCATION["name"]),
            float(st.session_state.get("loc_lat", config.DEFAULT_LOCATION["latitude"])),
            float(st.session_state.get("loc_lon", config.DEFAULT_LOCATION["longitude"])))


def plot_forecast(window: pd.DataFrame, units: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax = axes[0]
    if "temperature_2m" in window:
        ax.plot(window["time"], window["temperature_2m"], color="tab:red", label=f"Temperature ({units.get('temperature_2m', '')})")
    ax.set_ylabel(f"Temperature ({units.get('temperature_2m', '')})")
    if "relative_humidity_2m" in window:
        ax2 = ax.twinx()
        ax2.plot(window["time"], window["relative_humidity_2m"], color="tab:blue", label="Humidity")
        ax2.set_ylabel(f"Relative humidity ({units.get('relative_humidity_2m', '')})")
    ax.set_title("Forecast temperature and humidity (next 24 h)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    if "precipitation" in window:
        ax.bar(window["time"], window["precipitation"], width=0.03, color="tab:blue")
    ax.set_ylabel(f"Precipitation ({units.get('precipitation', '')})")
    if "precipitation_probability" in window:
        ax3 = ax.twinx()
        ax3.plot(window["time"], window["precipitation_probability"], color="tab:gray", linestyle="--")
        ax3.set_ylabel(f"Probability ({units.get('precipitation_probability', '%')})")
        ax3.set_ylim(0, 100)
    ax.set_title("Forecast precipitation amount and probability")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    show_fig(fig)


def page_weather() -> None:
    st.header("Weather Forecast")
    name, lat, lon = location()
    refresh = st.button("Refresh forecast (ignore cache)")
    weather, err = fetch_weather(lat, lon, refresh)
    if weather is None:
        st.error(f"Weather forecast unavailable: {err}")
        st.info("Offline soil-moisture predictions still work. Check your connection, the coordinates in the sidebar, and try again.")
        return
    st.caption(f"**FORECAST (future values, not observations)** for {name} - grid cell "
               f"{weather.latitude:.3f}, {weather.longitude:.3f} - timezone {weather.timezone} - fetched "
               f"{weather.fetched_at_utc} UTC{' (from local cache)' if weather.from_cache else ''}. Source: Open-Meteo.")
    window = next_hours(weather, 24)
    if window.empty:
        st.warning("The forecast does not cover the next 24 hours at this location.")
        return
    if weather.missing_variables:
        st.warning(f"These forecast variables were not returned and are omitted: {weather.missing_variables}")
    first = window.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    u = weather.units
    c1.metric("Forecast hour (local)", str(first["time"])[:16])
    if "temperature_2m" in window:
        c2.metric("Temperature", f"{first['temperature_2m']:.1f} {u.get('temperature_2m', '')}")
    if "relative_humidity_2m" in window:
        c3.metric("Relative humidity", f"{first['relative_humidity_2m']:.0f} {u.get('relative_humidity_2m', '')}")
    outlook = summarize_rainfall(window, 24)
    c4.metric("Rain next 24 h", f"{outlook.total_mm:.1f} {u.get('precipitation', 'mm')}" if outlook.total_mm is not None else "unknown")
    if outlook.max_probability_pct is not None:
        st.write(f"Maximum precipitation probability in the next 24 h: **{outlook.max_probability_pct:.0f}%**")
    for n in outlook.notes:
        st.warning(n)
    plot_forecast(window, u)
    with st.expander("Hourly forecast table"):
        st.dataframe(window)
        st.caption("Units: " + ", ".join(f"{k}: {v}" for k, v in u.items()))

    with st.expander("Historical observations (past reanalysis, NOT a forecast)"):
        end = date.today() - timedelta(days=7)
        start = end - timedelta(days=2)
        st.caption(f"Open-Meteo's archive lags real time by several days, so this shows {start} to {end} as an example of past data.")
        if st.button("Load historical example"):
            try:
                hist = get_historical(lat, lon, start, end)
                st.dataframe(hist.data)
                st.caption("HISTORICAL data (reanalysis). Units: " + ", ".join(f"{k}: {v}" for k, v in hist.units.items()))
            except WeatherServiceError as exc:
                st.error(f"Historical data unavailable: {exc}")


# ----------------------------------------------------------------------------
# D. Recommendation
# ----------------------------------------------------------------------------
def page_recommendation() -> None:
    st.header("Irrigation Recommendation")
    st.warning(config.CALIBRATION_NOTICE)
    pred = st.session_state.get("prediction")
    pid = st.session_state.get("prediction_id", 0)
    meta = load_metadata() or {}
    suggested = meta.get("suggested_thresholds", {})

    st.subheader("Inputs")
    c1, c2 = st.columns(2)
    predicted = c1.number_input("Predicted soil moisture after 24 h", value=float(pred.predicted) if pred else None,
                                format="%.4f", key=f"rec_pred_{pid}", placeholder="run a prediction first or type a value")
    current = c2.number_input("Current soil moisture (optional)", value=float(pred.current) if pred and pred.current is not None else None,
                              format="%.4f", key=f"rec_cur_{pid}", placeholder="optional")
    if pred is None:
        st.info("No prediction in this session yet. Use *Soil-Moisture Prediction*, or enter values manually.")

    st.subheader("Thresholds (you must calibrate these)")
    t1, t2 = st.columns(2)
    dry = t1.number_input("'Dry' threshold", min_value=0.0, value=suggested.get("dry_p20"), format="%.4f", key="thr_dry",
                          placeholder="required", help="Soil moisture below this is considered too dry.")
    crit = t2.number_input("'Critical' threshold (optional)", min_value=0.0, value=suggested.get("critical_p05"), format="%.4f",
                           key="thr_crit", placeholder="optional", help="Below this, the app will not advise waiting for rain.")
    if suggested:
        st.caption("Pre-filled values are the 20th and 5th percentiles of the TRAINING soil-moisture readings - an "
                   "illustration, NOT agronomic thresholds.")
    r1, r2, r3 = st.columns(3)
    window_h = r1.slider("Rain window (hours)", 6, 48, config.RAIN_WINDOW_HOURS)
    amount = r2.number_input("Rain amount that counts as meaningful (mm)", min_value=0.0, value=float(config.RAIN_AMOUNT_THRESHOLD_MM))
    prob = r3.number_input("Rain probability that counts (%)", min_value=0.0, max_value=100.0, value=float(config.RAIN_PROBABILITY_THRESHOLD_PCT))

    name, lat, lon = location()
    weather, werr = fetch_weather(lat, lon)
    rainfall = None
    if weather is None:
        st.error(f"Weather forecast unavailable ({werr}). The recommendation is made without it and rain is not assumed.")
    else:
        rainfall = summarize_rainfall(next_hours(weather, window_h), window_h)
        st.caption(f"Forecast for {name} (Open-Meteo, fetched {weather.fetched_at_utc} UTC).")

    try:
        thresholds = Thresholds(dry=dry, critical=crit, rain_amount_mm=amount, rain_probability_pct=prob)
        result = recommend(predicted, current, thresholds, rainfall)
    except ValueError as exc:
        st.error(f"Invalid thresholds: {exc}")
        return

    st.subheader("Recommendation")
    box = {Recommendation.MONITOR: st.success, Recommendation.IRRIGATION_MAY_BE_NEEDED: st.warning,
           Recommendation.DELAY_FOR_RAIN: st.info, Recommendation.INSUFFICIENT_INFORMATION: st.error}[result.code]
    box(f"**{result.headline}**")
    st.markdown("**Why:**")
    for line in result.reasons:
        st.markdown(f"- {line}")
    for w in result.warnings:
        st.warning(w)
    st.caption("No water quantity is suggested and no equipment is controlled. " + result.notice)


# ----------------------------------------------------------------------------
# E. Dataset analysis
# ----------------------------------------------------------------------------
def page_dataset() -> None:
    st.header("Dataset Analysis")
    ds, err = get_dataset()
    if ds is None:
        st.warning(err)
        st.info("See data/README.md for the download link and placement instructions, then run `python -m src.data_loader`.")
        return
    df, mapping, report, interval = ds
    st.caption(f"File: {report['file']}  |  cleaned rows: {report['rows_out']} (from {report['rows_in']})  |  "
               f"{report['start']} to {report['end']}  |  sampling interval: {interval}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns used", len(df.columns))
    c3.metric("Duplicate timestamps merged", report["duplicate_timestamps_merged"])
    st.write("**Detected column roles** (verify against the dataset's data dictionary): "
             + ", ".join(f"{r} -> `{c}`" for r, c in mapping.items() if c))

    st.subheader("Summary statistics")
    st.dataframe(df.describe().T)

    st.subheader("Missing values")
    miss = df.isna().sum().rename("missing").to_frame()
    miss["percent"] = 100 * miss["missing"] / max(len(df), 1)
    st.dataframe(miss)
    fig, ax = plt.subplots(figsize=(7, 2.5 + 0.2 * len(miss)))
    ax.barh(miss.index, miss["percent"])
    ax.set_xlabel("% missing")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    show_fig(fig)

    st.subheader("Distributions")
    cols = list(df.columns)
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col].dropna(), bins=40)
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    show_fig(fig)

    sm = mapping["soil_moisture"]
    st.subheader("Soil-moisture trend")
    daily = df.resample("D").agg({c: ("sum" if c == mapping.get("precipitation") else "mean") for c in df.columns})
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(df.index, df[sm], linewidth=0.4, alpha=0.5, label="Raw")
    ax.plot(daily.index, daily[sm], linewidth=1.5, label="Daily mean")
    ax.set_ylabel(sm)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    show_fig(fig)

    st.subheader("Relationships (daily aggregates)")
    pairs = [(r, mapping.get(r)) for r in ("temperature", "precipitation") if mapping.get(r)]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
        for ax, (role, col) in zip(axes[0], pairs):
            sub = daily[[col, sm]].dropna()
            ax.scatter(sub[col], sub[sm], s=12, alpha=0.6)
            ax.set_xlabel(f"{col} ({'daily total' if role == 'precipitation' else 'daily mean'})")
            ax.set_ylabel(f"{sm} (daily mean)")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        show_fig(fig)
    else:
        st.info("No temperature or precipitation columns were detected, so relationships cannot be shown.")
    st.write("Correlation of daily aggregates:")
    st.dataframe(daily.corr().round(3))
    st.caption("Correlation is not causation; irrigation events and rainfall both raise soil moisture.")


# ----------------------------------------------------------------------------
# F. Model evaluation
# ----------------------------------------------------------------------------
def page_evaluation() -> None:
    st.header("Model Evaluation")
    meta = load_metadata()
    if meta is None:
        st.warning("No evaluation results yet. Evaluation is PENDING until you run `python -m src.train_model` with the real dataset.")
        return
    table = comparison_table(meta["metrics"])
    st.subheader("Model comparison")
    st.dataframe(table.round(4))
    st.write(f"Selected by lowest **validation** RMSE: **{meta['best_model']}**. Test columns are held-out, later-in-time data.")
    split = meta.get("split", {})
    st.caption("Chronological split: " + "; ".join(f"{k}: {v['rows']} rows ({v['start'][:10]} to {v['end'][:10]})" for k, v in split.items()))

    pred_path = config.REPORTS_DIR / config.TEST_PREDICTIONS_FILENAME
    unit = f"({meta.get('soil_moisture_unit', '')})"
    if pred_path.exists():
        preds = pd.read_csv(pred_path, parse_dates=["timestamp"])
        st.subheader("Actual vs predicted (test set)")
        show_fig(plot_actual_vs_predicted_time(preds["timestamp"], preds["actual"], preds["predicted"], unit=unit))
        show_fig(plot_actual_vs_predicted_scatter(preds["actual"], preds["predicted"], unit=unit))
        st.subheader("Residuals")
        show_fig(plot_residuals(preds["timestamp"], preds["actual"], preds["predicted"]))
    else:
        st.info("Test predictions file not found; re-run training to regenerate plots.")

    if meta.get("feature_importance"):
        st.subheader("Feature importance (permutation, validation set)")
        show_fig(plot_feature_importance(pd.DataFrame(meta["feature_importance"])))
        st.caption("Importance = increase in validation RMSE when a feature's values are shuffled. Correlated features share importance.")

    st.subheader("Interpretation of these results")
    best = meta["best_model"]
    base = meta["metrics"].get("Persistence baseline", {}).get("test")
    mine = meta["metrics"][best]["test"]
    if base:
        change = 100 * (base["RMSE"] - mine["RMSE"]) / base["RMSE"] if base["RMSE"] else float("nan")
        if mine["RMSE"] < base["RMSE"]:
            st.write(f"On the test set, {best} has a **{change:.1f}% lower RMSE** than the persistence baseline "
                     f"({mine['RMSE']:.4g} vs {base['RMSE']:.4g}).")
        else:
            st.write(f"On the test set, {best} does **not** beat the persistence baseline "
                     f"(RMSE {mine['RMSE']:.4g} vs {base['RMSE']:.4g}); the extra features add no measurable value here.")
    if mine["R2"] == mine["R2"] and mine["R2"] < 0:
        st.write("A negative R² means the model does worse than always predicting the test-set mean.")
    st.caption("Results come from a single chronological split of one dataset (one site/season); they may not generalise to other fields.")


# ----------------------------------------------------------------------------
# G. Project information
# ----------------------------------------------------------------------------
def page_info() -> None:
    st.header("Project Information")
    with st.expander("Problem statement", expanded=True):
        st.write("Irrigating on a fixed schedule wastes water or stresses crops. Knowing how wet the soil will be tomorrow, "
                 "and whether rain is coming, supports better irrigation decisions.")
    with st.expander("Objectives"):
        st.write("Forecast soil moisture 24 h ahead; compare simple baselines with ensemble models; use a public weather API; "
                 "provide an explainable recommendation; keep the design ready for future IoT integration.")
    with st.expander("Methodology"):
        st.write("Chronological split with a purge gap, leakage-safe lag/rolling features, preprocessing fitted on training data only, "
                 "model selection on validation RMSE, final evaluation on held-out test data. See docs/methodology.md.")
    with st.expander("Dataset details"):
        st.write(DATASET_CITATION)
        st.write("Per the dataset description: an IoT-instrumented farm in Arnesano (Lecce, Italy), February-September 2025, "
                 "five sectors, 10-minute time series with ERA5 reanalysis weather. See data/README.md for the subset used.")
    with st.expander("Algorithms"):
        st.write("Persistence baseline, Linear Regression (standardised), Random Forest Regressor, Gradient Boosting Regressor (scikit-learn).")
    with st.expander("Results"):
        meta = load_metadata()
        if meta is None:
            st.write("**Pending** - no model has been trained yet.")
        else:
            st.dataframe(comparison_table(meta["metrics"]).round(4))
    with st.expander("Limitations"):
        st.markdown(
            "- One farm, one season: results may not transfer to other soils, crops or climates.\n"
            "- The model is trained on past weather; forecasts are used only by the recommendation rules.\n"
            "- Recommendation thresholds are not agronomically validated and need crop/soil calibration.\n"
            "- Weather forecasts are uncertain; the model does not know about future rain or irrigation.\n"
            "- Sensor drift, dropouts and unit differences between sensors are not modelled.")
    with st.expander("Future scope"):
        st.write("Integrate live readings from a separate IoT project (see `predict_from_history`), train with forecast-like weather "
                 "inputs, per-crop calibration, uncertainty intervals, multi-sector models.")
    with st.expander("References"):
        st.markdown(
            f"- {DATASET_CITATION}\n"
            "- Zippenfenig, P. (2023). Open-Meteo.com Weather API. https://open-meteo.com\n"
            "- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.\n"
            "- Breiman, L. (2001). Random Forests. Machine Learning 45(1), 5-32.\n"
            "- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189-1232.\n"
            "- Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999-2049.")


def main() -> None:
    page = sidebar()
    {
        "Overview": page_overview,
        "Soil-Moisture Prediction": page_prediction,
        "Weather Forecast": page_weather,
        "Irrigation Recommendation": page_recommendation,
        "Dataset Analysis": page_dataset,
        "Model Evaluation": page_evaluation,
        "Project Information": page_info,
    }[page]()


main()
