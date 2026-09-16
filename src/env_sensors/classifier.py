"""Good / moderate / bad classification for environmental sensor readings.

Pure functions, driven entirely by the ``environmental_sensors.thresholds``
section of ``configs/config.yaml`` -- no thresholds are hard-coded here, so
retuning the risk logic never requires touching this file.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional

Rating = Literal["good", "moderate", "bad", "unknown"]

_SEVERITY = {"good": 0, "moderate": 1, "bad": 2}


def classify_metric(value: Optional[float], rule: Dict[str, Any]) -> Rating:
    """Classify a single reading against one threshold rule.

    ``rule["direction"]`` is one of:
      - ``"high_bad"``: value <= good_limit -> good; <= moderate_limit -> moderate; else bad.
      - ``"low_bad"``:  value >= good_limit -> good; >= moderate_limit -> moderate; else bad.

    A missing/non-numeric/NaN value returns ``"unknown"`` rather than being
    guessed as good or bad.
    """
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(v):
        return "unknown"

    direction = rule["direction"]
    good_limit = rule["good_limit"]
    moderate_limit = rule["moderate_limit"]

    if direction == "high_bad":
        if v <= good_limit:
            return "good"
        if v <= moderate_limit:
            return "moderate"
        return "bad"
    if direction == "low_bad":
        if v >= good_limit:
            return "good"
        if v >= moderate_limit:
            return "moderate"
        return "bad"
    raise ValueError(f"Unknown threshold direction: {direction!r}")


def classify_snapshot(
    readings: Dict[str, Optional[float]],
    thresholds: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify every metric named in ``thresholds`` using ``readings``.

    Returns ``{"per_metric": {name: rating}, "overall": rating}``. The
    overall verdict is the worst rating among metrics with a known reading;
    a sensor that failed to report does not move the verdict either way.
    """
    per_metric: Dict[str, Rating] = {
        name: classify_metric(readings.get(name), rule)
        for name, rule in thresholds.items()
    }

    known = [r for r in per_metric.values() if r != "unknown"]
    overall: Rating = max(known, key=lambda r: _SEVERITY[r]) if known else "unknown"

    return {"per_metric": per_metric, "overall": overall}
