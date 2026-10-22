"""Open-Meteo weather client (forecast + historical), with caching and safe failure.

Forecast data   -> https://api.open-meteo.com/v1/forecast        (FUTURE, predicted)
Historical data -> https://archive-api.open-meteo.com/v1/archive (PAST, reanalysis)

No API key is required for these public endpoints. Every public function either
returns real API data or raises ``WeatherServiceError``; nothing is ever
fabricated. Callers should catch ``WeatherServiceError`` so that offline ML
predictions keep working when the network is down.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config


class WeatherServiceError(Exception):
    """Base error for weather retrieval problems (network, HTTP, malformed data)."""


class InvalidLocationError(WeatherServiceError, ValueError):
    """Latitude/longitude are missing or outside valid ranges."""


@dataclass
class WeatherData:
    kind: str                      # "forecast" (future) or "historical" (past observations/reanalysis)
    latitude: float
    longitude: float
    timezone: str
    utc_offset_seconds: int
    units: dict[str, str]
    data: pd.DataFrame             # column "time" (location-local, naive) + variables
    fetched_at_utc: str
    source_url: str
    from_cache: bool = False
    missing_variables: list[str] = field(default_factory=list)


@dataclass
class RainfallOutlook:
    """Summary of expected precipitation over a window (all fields may be None = unknown)."""

    window_hours: int
    total_mm: float | None
    max_probability_pct: float | None
    max_hourly_mm: float | None
    hours_covered: int
    available: bool
    notes: list[str] = field(default_factory=list)


def validate_location(latitude: Any, longitude: Any) -> tuple[float, float]:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise InvalidLocationError("Latitude and longitude must be numbers.") from exc
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise InvalidLocationError("Latitude and longitude must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise InvalidLocationError(f"Latitude {lat} is outside [-90, 90].")
    if not -180.0 <= lon <= 180.0:
        raise InvalidLocationError(f"Longitude {lon} is outside [-180, 180].")
    return lat, lon


# --------------------------------------------------------------------------
# Cache + HTTP
# --------------------------------------------------------------------------
def _cache_path(url: str, params: dict[str, Any], cache_dir: Path) -> Path:
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    return cache_dir / "weather" / f"{key}.json"


def _read_cache(path: Path, ttl: int) -> dict[str, Any] | None:
    try:
        if path.exists() and time.time() - path.stat().st_mtime <= ttl:
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # caching is best-effort only


def _get_json(url: str, params: dict[str, Any], cache_dir: Path, ttl: int, use_cache: bool,
              session: requests.Session | None = None) -> tuple[dict[str, Any], bool]:
    """GET ``url`` with retries/timeouts. Returns (payload, from_cache)."""
    path = _cache_path(url, params, cache_dir)
    if use_cache:
        cached = _read_cache(path, ttl)
        if cached is not None:
            return cached, True

    getter = session.get if session is not None else requests.get
    last_error: Exception | None = None
    for attempt in range(config.HTTP_RETRIES + 1):
        try:
            resp = getter(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            time.sleep(min(0.5 * (attempt + 1), 2.0)) if attempt < config.HTTP_RETRIES else None
            continue
        except requests.RequestException as exc:
            raise WeatherServiceError(f"Weather request failed: {exc}") from exc

        if resp.status_code >= 500:
            last_error = WeatherServiceError(f"Weather service returned HTTP {resp.status_code}.")
            time.sleep(min(0.5 * (attempt + 1), 2.0)) if attempt < config.HTTP_RETRIES else None
            continue
        if resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("reason", "")
            except Exception:  # noqa: BLE001
                pass
            raise WeatherServiceError(f"Weather service returned HTTP {resp.status_code}. {detail}".strip())
        try:
            payload = resp.json()
        except ValueError as exc:
            raise WeatherServiceError("Weather service returned a response that is not valid JSON.") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise WeatherServiceError(f"Weather service error: {payload.get('reason', 'unknown reason')}")
        if use_cache:
            _write_cache(path, payload)
        return payload, False

    if isinstance(last_error, WeatherServiceError):
        raise last_error
    raise WeatherServiceError(
        f"Could not reach the weather service ({type(last_error).__name__}). Check your internet connection."
    ) from last_error


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_hourly_payload(payload: dict[str, Any], requested: tuple[str, ...], kind: str, url: str,
                         from_cache: bool = False) -> WeatherData:
    """Turn an Open-Meteo JSON payload into ``WeatherData``; tolerant of missing variables."""
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise WeatherServiceError("Weather response did not contain hourly data.")
    times = pd.to_datetime(hourly["time"], errors="coerce")
    if times.isna().all():
        raise WeatherServiceError("Weather response contained no valid timestamps.")
    frame = pd.DataFrame({"time": times})
    missing: list[str] = []
    for var in requested:
        values = hourly.get(var)
        if values is None or len(values) != len(frame):
            missing.append(var)
            continue
        frame[var] = pd.to_numeric(pd.Series(values), errors="coerce")
        if frame[var].isna().all():
            missing.append(var)
    frame = frame.dropna(subset=["time"]).reset_index(drop=True)
    try:
        lat, lon = float(payload["latitude"]), float(payload["longitude"])
    except (KeyError, TypeError, ValueError):
        raise WeatherServiceError("Weather response did not include the grid-cell coordinates.") from None
    return WeatherData(
        kind=kind,
        latitude=lat,
        longitude=lon,
        timezone=str(payload.get("timezone", "UTC")),
        utc_offset_seconds=int(payload.get("utc_offset_seconds", 0) or 0),
        units={k: str(v) for k, v in (payload.get("hourly_units") or {}).items()},
        data=frame,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_url=url,
        from_cache=from_cache,
        missing_variables=missing,
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_forecast(latitude: float, longitude: float, forecast_days: int = 3, use_cache: bool = True,
                 cache_dir: Path | None = None, session: requests.Session | None = None) -> WeatherData:
    """Hourly FORECAST (future values) for a location."""
    lat, lon = validate_location(latitude, longitude)
    if not 1 <= int(forecast_days) <= 16:
        raise ValueError("forecast_days must be between 1 and 16.")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(config.FORECAST_HOURLY_VARIABLES),
        "forecast_days": int(forecast_days),
        "timezone": "auto",
    }
    payload, cached = _get_json(config.OPEN_METEO_FORECAST_URL, params, Path(cache_dir) if cache_dir else config.CACHE_DIR,
                                config.FORECAST_CACHE_TTL_SECONDS, use_cache, session)
    return parse_hourly_payload(payload, config.FORECAST_HOURLY_VARIABLES, "forecast", config.OPEN_METEO_FORECAST_URL, cached)


def get_historical(latitude: float, longitude: float, start_date: date, end_date: date, use_cache: bool = True,
                   cache_dir: Path | None = None, session: requests.Session | None = None) -> WeatherData:
    """Hourly HISTORICAL data (past reanalysis, not a forecast) for a date range.

    The archive typically lags real time by a few days, so recent dates may be empty.
    """
    lat, lon = validate_location(latitude, longitude)
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date.")
    if end_date > date.today():
        raise ValueError("Historical data cannot extend into the future; use get_forecast() instead.")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(config.ARCHIVE_HOURLY_VARIABLES),
        "timezone": "auto",
    }
    payload, cached = _get_json(config.OPEN_METEO_ARCHIVE_URL, params, Path(cache_dir) if cache_dir else config.CACHE_DIR,
                                config.ARCHIVE_CACHE_TTL_SECONDS, use_cache, session)
    return parse_hourly_payload(payload, config.ARCHIVE_HOURLY_VARIABLES, "historical", config.OPEN_METEO_ARCHIVE_URL, cached)


def local_now(weather: WeatherData, now_utc: datetime | None = None) -> pd.Timestamp:
    """Current time at the forecast location (naive), using the API-reported UTC offset."""
    now_utc = now_utc or datetime.now(timezone.utc)
    return pd.Timestamp((now_utc + timedelta(seconds=weather.utc_offset_seconds)).replace(tzinfo=None))


def next_hours(weather: WeatherData, hours: int = 24, now_utc: datetime | None = None) -> pd.DataFrame:
    """Rows from the current hour onward (at the location) covering at most ``hours`` hours."""
    if weather.kind != "forecast":
        raise ValueError("next_hours() only applies to forecast data.")
    start = local_now(weather, now_utc).floor("h")
    df = weather.data
    sel = df[(df["time"] >= start) & (df["time"] < start + pd.Timedelta(hours=hours))]
    return sel.reset_index(drop=True)


def summarize_rainfall(window: pd.DataFrame | None, window_hours: int = 24) -> RainfallOutlook:
    """Summarise expected precipitation; unknown fields stay None (never assumed to be zero)."""
    if window is None or window.empty:
        return RainfallOutlook(window_hours, None, None, None, 0, False, ["No forecast rows cover the requested window."])
    notes: list[str] = []
    total = max_hourly = max_prob = None
    if "precipitation" in window and window["precipitation"].notna().any():
        p = window["precipitation"].dropna()
        total, max_hourly = float(p.sum()), float(p.max())
        if len(p) < len(window):
            notes.append("Some hourly precipitation values were missing; the total may be underestimated.")
    else:
        notes.append("Precipitation amount is not available in the forecast.")
    if "precipitation_probability" in window and window["precipitation_probability"].notna().any():
        max_prob = float(window["precipitation_probability"].dropna().max())
    else:
        notes.append("Precipitation probability is not available in the forecast.")
    return RainfallOutlook(window_hours, total, max_prob, max_hourly, int(len(window)),
                           total is not None or max_prob is not None, notes)
