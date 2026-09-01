"""Unit tests for environmental-sensor good/moderate/bad classification."""
from __future__ import annotations

import math

import pytest

from src.env_sensors.classifier import classify_metric, classify_snapshot

HIGH_BAD = {"direction": "high_bad", "good_limit": 60, "moderate_limit": 85}
LOW_BAD = {"direction": "low_bad", "good_limit": 700, "moderate_limit": 400}


# ---------------------------------------------------------------------------
# classify_metric -- boundary correctness for both directions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "good"),
        (60, "good"),          # exactly good_limit -> good (inclusive)
        (60.0001, "moderate"),
        (85, "moderate"),      # exactly moderate_limit -> moderate (inclusive)
        (85.0001, "bad"),
        (200, "bad"),
    ],
)
def test_classify_metric_high_bad(value, expected):
    assert classify_metric(value, HIGH_BAD) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1000, "good"),
        (700, "good"),         # exactly good_limit -> good (inclusive)
        (699.9999, "moderate"),
        (400, "moderate"),     # exactly moderate_limit -> moderate (inclusive)
        (399.9999, "bad"),
        (0, "bad"),
    ],
)
def test_classify_metric_low_bad(value, expected):
    assert classify_metric(value, LOW_BAD) == expected


@pytest.mark.parametrize("value", [None, float("nan"), "not-a-number"])
def test_classify_metric_missing_value_is_unknown(value):
    assert classify_metric(value, HIGH_BAD) == "unknown"


def test_classify_metric_rejects_unknown_direction():
    with pytest.raises(ValueError):
        classify_metric(10, {"direction": "sideways", "good_limit": 1, "moderate_limit": 2})


# ---------------------------------------------------------------------------
# classify_snapshot -- overall verdict is the worst known rating
# ---------------------------------------------------------------------------
THRESHOLDS = {"humidity_pct": HIGH_BAD, "air_quality_raw": LOW_BAD}


def test_classify_snapshot_all_good():
    result = classify_snapshot({"humidity_pct": 40, "air_quality_raw": 900}, THRESHOLDS)
    assert result["per_metric"] == {"humidity_pct": "good", "air_quality_raw": "good"}
    assert result["overall"] == "good"


def test_classify_snapshot_worst_of_mixed_ratings():
    result = classify_snapshot({"humidity_pct": 40, "air_quality_raw": 100}, THRESHOLDS)
    assert result["per_metric"]["air_quality_raw"] == "bad"
    assert result["overall"] == "bad"


def test_classify_snapshot_unknown_metric_does_not_shift_overall():
    result = classify_snapshot({"humidity_pct": 70, "air_quality_raw": None}, THRESHOLDS)
    assert result["per_metric"]["air_quality_raw"] == "unknown"
    assert result["overall"] == "moderate"  # driven only by the known humidity reading


def test_classify_snapshot_all_unknown_is_unknown():
    result = classify_snapshot({"humidity_pct": None, "air_quality_raw": math.nan}, THRESHOLDS)
    assert result["overall"] == "unknown"
