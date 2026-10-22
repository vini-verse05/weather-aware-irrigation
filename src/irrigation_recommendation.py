"""Rule-based irrigation decision support, deliberately separate from the ML model.

The rules combine (a) the predicted soil moisture 24 h ahead, (b) the current
reading when available and (c) the rainfall outlook from the weather forecast.

IMPORTANT
* Thresholds are user-configurable inputs. No universal threshold is correct for
  every crop or soil; they need crop- and soil-specific calibration.
* The module never computes water quantities and never controls equipment.
* Missing information is handled conservatively: rain is NEVER assumed when the
  forecast is unavailable, and with no thresholds the answer is
  "insufficient information".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from . import config
from .weather_service import RainfallOutlook


class Recommendation(str, Enum):
    MONITOR = "monitor"
    IRRIGATION_MAY_BE_NEEDED = "irrigation_may_be_needed"
    DELAY_FOR_RAIN = "delay_for_rain"
    INSUFFICIENT_INFORMATION = "insufficient_information"


HEADLINES = {
    Recommendation.MONITOR: "Monitor soil moisture - no irrigation indicated right now.",
    Recommendation.IRRIGATION_MAY_BE_NEEDED: "Irrigation may be needed.",
    Recommendation.DELAY_FOR_RAIN: "Consider delaying irrigation - meaningful rainfall is forecast.",
    Recommendation.INSUFFICIENT_INFORMATION: "Insufficient information to make a reliable recommendation.",
}


@dataclass
class Thresholds:
    """User-configurable decision thresholds (same unit as the soil-moisture values)."""

    dry: float | None = None                 # below this, soil is considered too dry
    critical: float | None = None            # optional: below this, do not wait for rain
    rain_amount_mm: float = config.RAIN_AMOUNT_THRESHOLD_MM
    rain_probability_pct: float = config.RAIN_PROBABILITY_THRESHOLD_PCT

    def validate(self) -> None:
        for name in ("dry", "critical"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
                raise ValueError(f"Threshold '{name}' must be a non-negative finite number or None.")
        if self.dry is not None and self.critical is not None and self.critical > self.dry:
            raise ValueError("The critical threshold must not be above the dry threshold.")
        if not (math.isfinite(self.rain_amount_mm) and self.rain_amount_mm >= 0):
            raise ValueError("rain_amount_mm must be a non-negative finite number.")
        if not (math.isfinite(self.rain_probability_pct) and 0 <= self.rain_probability_pct <= 100):
            raise ValueError("rain_probability_pct must be between 0 and 100.")


@dataclass
class RecommendationResult:
    code: Recommendation
    headline: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    notice: str = config.CALIBRATION_NOTICE


def _valid(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def assess_rain(rainfall: RainfallOutlook | None, thresholds: Thresholds) -> tuple[str, list[str]]:
    """Return ("significant" | "low" | "unknown", explanation lines)."""
    if rainfall is None or not rainfall.available:
        return "unknown", ["No usable rainfall forecast is available, so rain is not assumed."]
    window = f"next {rainfall.window_hours} h"
    if rainfall.total_mm is None:
        extra = f" (max probability {rainfall.max_probability_pct:.0f}%)" if rainfall.max_probability_pct is not None else ""
        return "unknown", [f"Forecast precipitation amount is unavailable for the {window}{extra}, so rain is not assumed."]
    prob = rainfall.max_probability_pct
    lines = [f"Forecast precipitation in the {window}: {rainfall.total_mm:.1f} mm"
             + (f", max probability {prob:.0f}%." if prob is not None else " (probability unavailable).")]
    amount_ok = rainfall.total_mm >= thresholds.rain_amount_mm
    prob_ok = prob is None or prob >= thresholds.rain_probability_pct
    if amount_ok and prob_ok:
        lines.append(f"This meets the rain rule (>= {thresholds.rain_amount_mm:g} mm and probability >= "
                     f"{thresholds.rain_probability_pct:g}% when known).")
        return "significant", lines
    lines.append(f"This is below the rain rule (>= {thresholds.rain_amount_mm:g} mm and probability >= "
                 f"{thresholds.rain_probability_pct:g}%).")
    return "low", lines


def recommend(
    predicted_sm: float | None,
    current_sm: float | None = None,
    thresholds: Thresholds | None = None,
    rainfall: RainfallOutlook | None = None,
) -> RecommendationResult:
    """Produce an explained recommendation. Raises ValueError for invalid thresholds."""
    thresholds = thresholds or Thresholds()
    thresholds.validate()
    pred_ok, cur_ok = _valid(predicted_sm), _valid(current_sm)
    details: dict[str, Any] = {"predicted_sm": predicted_sm if pred_ok else None,
                               "current_sm": current_sm if cur_ok else None,
                               "thresholds": {"dry": thresholds.dry, "critical": thresholds.critical,
                                              "rain_amount_mm": thresholds.rain_amount_mm,
                                              "rain_probability_pct": thresholds.rain_probability_pct}}

    def result(code: Recommendation, reasons: list[str], warnings: list[str]) -> RecommendationResult:
        return RecommendationResult(code, HEADLINES[code], reasons, warnings, details)

    if not pred_ok and not cur_ok:
        return result(Recommendation.INSUFFICIENT_INFORMATION,
                      ["Neither a predicted nor a current soil-moisture value is available."], [])
    if thresholds.dry is None:
        return result(Recommendation.INSUFFICIENT_INFORMATION,
                      ["No 'dry' moisture threshold has been set, so soil moisture cannot be judged."],
                      ["Set a threshold calibrated for your crop and soil."])

    warnings: list[str] = []
    reasons: list[str] = []
    if not pred_ok:
        warnings.append("No 24 h prediction available; the decision uses the current reading only.")
    if not cur_ok:
        warnings.append("No current reading supplied; the decision uses the predicted value only.")

    dry_pred = pred_ok and predicted_sm < thresholds.dry
    dry_now = cur_ok and current_sm < thresholds.dry
    if pred_ok:
        reasons.append(f"Predicted soil moisture in 24 h: {predicted_sm:.3g} "
                       f"({'below' if dry_pred else 'at or above'} the dry threshold {thresholds.dry:g}).")
    if cur_ok:
        reasons.append(f"Current soil moisture: {current_sm:.3g} "
                       f"({'below' if dry_now else 'at or above'} the dry threshold {thresholds.dry:g}).")

    if not (dry_pred or dry_now):
        reasons.append("Soil moisture is not below the dry threshold now or in 24 h.")
        details["rain_assessment"] = "not evaluated"
        return result(Recommendation.MONITOR, reasons, warnings)

    if dry_pred and cur_ok and not dry_now:
        reasons.append("Soil is expected to dry below the threshold within 24 h.")
    if dry_now and pred_ok and not dry_pred:
        reasons.append("Soil is currently below the threshold even though the model expects recovery "
                       "(the model does not know about the rainfall forecast).")

    # Critically dry? Use the current reading if known, otherwise the prediction.
    reference = current_sm if cur_ok else predicted_sm
    critical = thresholds.critical is not None and reference < thresholds.critical
    if critical:
        reasons.append(f"Moisture is below the critical threshold {thresholds.critical:g}; waiting for rain is not advised.")

    rain, rain_lines = assess_rain(rainfall, thresholds)
    reasons.extend(rain_lines)
    details["rain_assessment"] = rain
    if rainfall is not None:
        warnings.extend(rainfall.notes)

    if rain == "significant" and not critical:
        reasons.append("Rainfall may replenish soil moisture, so irrigation could be postponed. "
                       "Re-check after the rain.")
        return result(Recommendation.DELAY_FOR_RAIN, reasons, warnings)
    if rain == "unknown":
        warnings.append("Check the local weather outlook before irrigating; this tool could not confirm rain.")
    return result(Recommendation.IRRIGATION_MAY_BE_NEEDED, reasons, warnings)
