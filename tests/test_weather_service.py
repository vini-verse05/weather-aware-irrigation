"""Weather service tests. All HTTP traffic is faked - no internet connection is used."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
import requests

from src import config, weather_service
from src.weather_service import (
    InvalidLocationError,
    WeatherServiceError,
    get_forecast,
    get_historical,
    next_hours,
    summarize_rainfall,
    validate_location,
)


class FakeResponse:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Returns queued responses/exceptions and records calls."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


def forecast_payload(hours=72, drop=(), nan_precip=False):
    times = [(datetime(2026, 10, 21) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]
    hourly = {
        "time": times,
        "temperature_2m": [15.0 + (i % 24) * 0.2 for i in range(hours)],
        "relative_humidity_2m": [70 for _ in range(hours)],
        "precipitation": [0.0] * hours,
        "precipitation_probability": [10] * hours,
        "et0_fao_evapotranspiration": [0.1] * hours,
        "wind_speed_10m": [8.0] * hours,
    }
    hourly["precipitation"][14] = 3.0
    hourly["precipitation_probability"][14] = 80
    if nan_precip:
        hourly["precipitation"] = [None] * hours
    for k in drop:
        hourly.pop(k)
    return {
        "latitude": 40.34, "longitude": 18.1, "timezone": "Europe/Rome", "utc_offset_seconds": 7200,
        "hourly_units": {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm",
                         "precipitation_probability": "%"},
        "hourly": hourly,
    }


NOW = datetime(2026, 10, 21, 10, 30, tzinfo=timezone.utc)  # 12:30 local (UTC+2)


def test_forecast_parsing_units_and_location(tmp_path):
    session = FakeSession(FakeResponse(200, forecast_payload()))
    w = get_forecast(40.33, 18.10, cache_dir=tmp_path, session=session)
    assert w.kind == "forecast"
    assert w.timezone == "Europe/Rome"
    assert w.units["precipitation"] == "mm"
    assert {"temperature_2m", "relative_humidity_2m", "precipitation", "precipitation_probability"} <= set(w.data.columns)
    assert w.missing_variables == []
    url, params, timeout = session.calls[0]
    assert url == config.OPEN_METEO_FORECAST_URL
    assert params["latitude"] == 40.33 and params["longitude"] == 18.10
    assert timeout == config.HTTP_TIMEOUT_SECONDS
    assert "apikey" not in params  # public endpoint, no key needed


def test_next_hours_starts_at_current_local_hour_and_spans_window(tmp_path):
    w = get_forecast(40.33, 18.10, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, forecast_payload())))
    window = next_hours(w, 24, now_utc=NOW)
    assert window["time"].iloc[0] == pd.Timestamp("2026-10-21 12:00")
    assert len(window) == 24
    assert window["time"].iloc[-1] == pd.Timestamp("2026-10-22 11:00")


def test_rainfall_summary_from_forecast(tmp_path):
    w = get_forecast(40.33, 18.10, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, forecast_payload())))
    outlook = summarize_rainfall(next_hours(w, 24, now_utc=NOW), 24)
    assert outlook.available
    assert outlook.total_mm == 3.0 and outlook.max_probability_pct == 80.0 and outlook.max_hourly_mm == 3.0


def test_missing_forecast_fields_are_reported_not_invented(tmp_path):
    payload = forecast_payload(drop=("precipitation_probability", "wind_speed_10m"))
    w = get_forecast(40.33, 18.10, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, payload)))
    assert set(w.missing_variables) == {"precipitation_probability", "wind_speed_10m"}
    outlook = summarize_rainfall(next_hours(w, 24, now_utc=NOW), 24)
    assert outlook.max_probability_pct is None  # unknown, never assumed to be 0
    assert any("probability" in n.lower() for n in outlook.notes)


def test_all_null_precipitation_means_unknown_not_zero(tmp_path):
    w = get_forecast(40.33, 18.10, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, forecast_payload(nan_precip=True))))
    assert "precipitation" in w.missing_variables
    outlook = summarize_rainfall(next_hours(w, 24, now_utc=NOW), 24)
    assert outlook.total_mm is None


def test_summarize_rainfall_handles_empty_window():
    outlook = summarize_rainfall(pd.DataFrame(), 24)
    assert not outlook.available and outlook.total_mm is None
    assert not summarize_rainfall(None, 24).available


def test_response_without_hourly_block_raises(tmp_path):
    with pytest.raises(WeatherServiceError, match="hourly"):
        get_forecast(40, 18, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, {"latitude": 1, "longitude": 1})))


def test_timeout_is_converted_to_service_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 0)
    session = FakeSession(requests.Timeout("slow"))
    with pytest.raises(WeatherServiceError, match="Could not reach"):
        get_forecast(40, 18, cache_dir=tmp_path, session=session)


def test_connection_error_is_converted_to_service_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 0)
    with pytest.raises(WeatherServiceError):
        get_forecast(40, 18, cache_dir=tmp_path, session=FakeSession(requests.ConnectionError("offline")))


def test_server_error_and_client_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 0)
    with pytest.raises(WeatherServiceError, match="HTTP 503"):
        get_forecast(40, 18, cache_dir=tmp_path, session=FakeSession(FakeResponse(503, {})))
    with pytest.raises(WeatherServiceError, match="HTTP 400"):
        get_forecast(40, 18, cache_dir=tmp_path, session=FakeSession(FakeResponse(400, {"reason": "bad param"})))


def test_invalid_json_and_error_payload(tmp_path):
    with pytest.raises(WeatherServiceError, match="JSON"):
        get_forecast(40, 18, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, bad_json=True)))
    with pytest.raises(WeatherServiceError, match="boom"):
        get_forecast(40, 18, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, {"error": True, "reason": "boom"})))


def test_retry_succeeds_after_transient_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    monkeypatch.setattr(weather_service.time, "sleep", lambda s: None)
    session = FakeSession(requests.ConnectionError("blip"), FakeResponse(200, forecast_payload()))
    w = get_forecast(40, 18, cache_dir=tmp_path, session=session)
    assert len(session.calls) == 2 and w.kind == "forecast"


@pytest.mark.parametrize("lat,lon", [(91, 0), (-91, 0), (0, 181), (0, -181), ("x", 0), (None, 0), (float("nan"), 0)])
def test_invalid_locations_are_rejected_before_any_request(lat, lon, tmp_path):
    session = FakeSession(FakeResponse(200, forecast_payload()))
    with pytest.raises(InvalidLocationError):
        get_forecast(lat, lon, cache_dir=tmp_path, session=session)
    assert session.calls == []


def test_validate_location_accepts_numeric_strings():
    assert validate_location("40.5", "18.1") == (40.5, 18.1)


def test_responses_are_cached(tmp_path):
    first = FakeSession(FakeResponse(200, forecast_payload()))
    w1 = get_forecast(40, 18, cache_dir=tmp_path, session=first)
    assert not w1.from_cache
    second = FakeSession(requests.ConnectionError("offline"))
    w2 = get_forecast(40, 18, cache_dir=tmp_path, session=second)  # served from cache, no network
    assert w2.from_cache and second.calls == []
    third = FakeSession(FakeResponse(200, forecast_payload()))
    get_forecast(40, 18, cache_dir=tmp_path, session=third, use_cache=False)
    assert len(third.calls) == 1


def test_historical_is_labelled_and_cannot_reach_into_the_future(tmp_path):
    payload = forecast_payload(hours=48)
    payload["hourly"].pop("precipitation_probability")
    end = date.today() - timedelta(days=10)
    w = get_historical(40, 18, end - timedelta(days=1), end, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, payload)))
    assert w.kind == "historical"
    assert w.source_url == config.OPEN_METEO_ARCHIVE_URL
    with pytest.raises(ValueError):
        get_historical(40, 18, date.today(), date.today() + timedelta(days=2), cache_dir=tmp_path)
    with pytest.raises(ValueError):
        get_historical(40, 18, end, end - timedelta(days=3), cache_dir=tmp_path)
    with pytest.raises(ValueError):
        next_hours(w, 24)  # next_hours is for forecasts only


def test_invalid_forecast_days_rejected(tmp_path):
    with pytest.raises(ValueError):
        get_forecast(40, 18, forecast_days=0, cache_dir=tmp_path, session=FakeSession(FakeResponse(200, forecast_payload())))
