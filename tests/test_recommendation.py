"""Irrigation recommendation rules (pure logic, no ML model or network needed)."""

import pytest

from src import config
from src.irrigation_recommendation import Recommendation, Thresholds, recommend
from src.weather_service import RainfallOutlook

T = Thresholds(dry=20.0, critical=10.0, rain_amount_mm=2.0, rain_probability_pct=50.0)


def rain(total=5.0, prob=80.0, available=True, hours=24):
    return RainfallOutlook(hours, total, prob, total, hours, available, [])


def test_no_threshold_means_insufficient_information():
    r = recommend(15.0, 18.0, Thresholds(dry=None), rain())
    assert r.code == Recommendation.INSUFFICIENT_INFORMATION
    assert any("threshold" in x.lower() for x in r.reasons)


def test_no_soil_moisture_values_means_insufficient_information():
    assert recommend(None, None, T, rain()).code == Recommendation.INSUFFICIENT_INFORMATION
    assert recommend(float("nan"), float("inf"), T, rain()).code == Recommendation.INSUFFICIENT_INFORMATION
    assert recommend(True, None, T, rain()).code == Recommendation.INSUFFICIENT_INFORMATION


def test_wet_soil_means_monitor_even_if_rain_is_forecast():
    r = recommend(30.0, 28.0, T, rain())
    assert r.code == Recommendation.MONITOR


def test_dry_soil_without_rain_means_irrigation_may_be_needed():
    r = recommend(15.0, 18.0, T, rain(total=0.0, prob=5.0))
    assert r.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert r.details["rain_assessment"] == "low"


def test_dry_soil_with_significant_rain_means_delay():
    r = recommend(15.0, 18.0, T, rain(total=6.0, prob=85.0))
    assert r.code == Recommendation.DELAY_FOR_RAIN
    assert any("mm" in x for x in r.reasons)


def test_rain_amount_without_probability_still_counts_when_amount_is_high():
    assert recommend(15.0, 18.0, T, rain(total=6.0, prob=None)).code == Recommendation.DELAY_FOR_RAIN


def test_high_probability_but_tiny_amount_does_not_delay():
    assert recommend(15.0, 18.0, T, rain(total=0.2, prob=90.0)).code == Recommendation.IRRIGATION_MAY_BE_NEEDED


def test_large_amount_but_low_probability_does_not_delay():
    assert recommend(15.0, 18.0, T, rain(total=6.0, prob=20.0)).code == Recommendation.IRRIGATION_MAY_BE_NEEDED


def test_critically_dry_soil_is_not_delayed_for_rain():
    r = recommend(8.0, 9.0, T, rain(total=10.0, prob=95.0))
    assert r.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert any("critical" in x.lower() for x in r.reasons)


def test_missing_weather_is_handled_conservatively():
    for outlook in (None, rain(available=False), RainfallOutlook(24, None, None, None, 0, False, ["x"])):
        r = recommend(15.0, 18.0, T, outlook)
        assert r.code == Recommendation.IRRIGATION_MAY_BE_NEEDED  # rain is never assumed
        assert r.details["rain_assessment"] == "unknown"
        assert any("weather" in w.lower() or "rain" in w.lower() for w in r.warnings)


def test_probability_only_forecast_does_not_justify_delay():
    partial = RainfallOutlook(24, None, 90.0, None, 24, True, [])
    r = recommend(15.0, 18.0, T, partial)
    assert r.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert r.details["rain_assessment"] == "unknown"


def test_prediction_only_and_current_only_inputs_work_with_warnings():
    only_pred = recommend(15.0, None, T, rain(total=0, prob=0))
    assert only_pred.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert any("No current reading" in w for w in only_pred.warnings)
    only_cur = recommend(None, 15.0, T, rain(total=0, prob=0))
    assert only_cur.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert any("No 24 h prediction" in w for w in only_cur.warnings)


def test_currently_dry_but_predicted_recovery_is_flagged_in_reasons():
    r = recommend(25.0, 15.0, T, rain(total=0, prob=0))
    assert r.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert any("expects recovery" in x for x in r.reasons)


def test_predicted_drying_from_wet_now_is_flagged():
    r = recommend(15.0, 30.0, T, rain(total=0, prob=0))
    assert r.code == Recommendation.IRRIGATION_MAY_BE_NEEDED
    assert any("expected to dry" in x for x in r.reasons)


@pytest.mark.parametrize("kwargs", [
    {"dry": -1.0},
    {"dry": float("nan")},
    {"dry": 10.0, "critical": 20.0},
    {"dry": 10.0, "rain_amount_mm": -1.0},
    {"dry": 10.0, "rain_probability_pct": 101.0},
    {"dry": 10.0, "rain_probability_pct": -5.0},
    {"dry": "wet"},
])
def test_invalid_thresholds_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        recommend(15.0, 15.0, Thresholds(**kwargs), rain())


def test_every_recommendation_explains_itself_and_carries_the_calibration_notice():
    cases = [
        recommend(None, None, T, None),
        recommend(30, 30, T, rain()),
        recommend(15, 15, T, rain(0, 0)),
        recommend(15, 15, T, rain(6, 90)),
        recommend(5, 5, T, rain(6, 90)),
        recommend(15, 15, Thresholds(dry=None), None),
    ]
    for r in cases:
        assert r.reasons, r.code
        assert r.headline
        assert r.notice == config.CALIBRATION_NOTICE
        assert "calibrat" in r.notice.lower()


def test_recommendation_never_contains_water_quantities_or_pump_commands():
    r = recommend(15.0, 15.0, T, rain(0, 0))
    text = " ".join([r.headline, *r.reasons, *r.warnings]).lower()
    for forbidden in ("litre", "liter", "gallon", "turn on", "activate pump", "start pump"):
        assert forbidden not in text
