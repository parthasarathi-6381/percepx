"""Unit tests for the ResolutionPolicy, focused on boundary correctness."""
from __future__ import annotations

import numpy as np
import pytest

from src.mapping.resolution_policy import ResolutionPolicy


@pytest.fixture
def policy():
    # Default SIH zones.
    return ResolutionPolicy(
        near_range=10.0, mid_range=30.0, far_range=60.0, max_range=100.0,
        res_near=0.05, res_mid=0.15, res_far=0.30, res_very_far=0.50,
    )


# ---------------------------------------------------------------------------
# Scalar zone selection at the exact boundaries (the critical cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "r,expected_zone,expected_res",
    [
        (0.0, "near", 0.05),
        (5.0, "near", 0.05),
        (10.0, "near", 0.05),        # exactly 10 -> near (inclusive)
        (10.0001, "mid", 0.15),
        (20.0, "mid", 0.15),
        (30.0, "mid", 0.15),         # exactly 30 -> mid
        (30.0001, "far", 0.30),
        (45.0, "far", 0.30),
        (60.0, "far", 0.30),         # exactly 60 -> far
        (60.0001, "very_far", 0.50),
        (80.0, "very_far", 0.50),
        (100.0, "very_far", 0.50),   # exactly 100 -> very_far
    ],
)
def test_zone_and_resolution_at_boundaries(policy, r, expected_zone, expected_res):
    assert policy.zone_for_distance(r) == expected_zone
    assert policy.resolution_for_distance(r) == expected_res


def test_beyond_max_range_is_discarded(policy):
    assert policy.zone_for_distance(100.0001) is None
    assert policy.zone_for_distance(150.0) is None
    assert policy.resolution_for_distance(200.0) is None


def test_negative_distance_raises(policy):
    with pytest.raises(ValueError):
        policy.zone_for_distance(-1.0)


# ---------------------------------------------------------------------------
# Vectorized assignment matches the scalar version, incl. boundaries
# ---------------------------------------------------------------------------
def test_assign_zones_vectorized_boundaries(policy):
    r = np.array([0.0, 10.0, 10.001, 30.0, 30.001, 60.0, 60.001, 100.0, 100.001])
    idx = policy.assign_zones(r)
    #            0     0      1      1      2      2      3       3      -1
    assert list(idx) == [0, 0, 1, 1, 2, 2, 3, 3, -1]


def test_assign_zones_matches_scalar(policy):
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 120, size=1000)
    idx = policy.assign_zones(r)
    names = ["near", "mid", "far", "very_far"]
    for ri, ii in zip(r, idx):
        scalar = policy.zone_for_distance(float(ri))
        if scalar is None:
            assert ii == -1
        else:
            assert names[ii] == scalar


def test_assign_zones_negative_raises(policy):
    with pytest.raises(ValueError):
        policy.assign_zones(np.array([1.0, -0.5]))


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------
def test_from_config():
    cfg = {
        "mapping": {"near_range": 5, "mid_range": 20, "far_range": 40, "max_range": 80},
        "resolution": {"near": 0.02, "mid": 0.1, "far": 0.2, "very_far": 0.4},
    }
    p = ResolutionPolicy.from_config(cfg)
    assert p.max_range == 80
    assert p.resolution_for_distance(4.0) == 0.02
    assert p.resolution_for_distance(80.0) == 0.4
    assert p.zone_for_distance(80.001) is None


def test_non_monotonic_ranges_raise():
    with pytest.raises(ValueError):
        ResolutionPolicy(near_range=30, mid_range=10, far_range=60, max_range=100)
