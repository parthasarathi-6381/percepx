"""Unit tests for the segmentation subpackage."""
from __future__ import annotations

import numpy as np
import pytest

from src.segmentation import (
    BaselineSegmenter,
    PretrainedSegmenter,
    SegmentationResult,
    build_segmenter,
    CLASS_ID,
)


# ---------------------------------------------------------------------------
# SegmentationResult container
# ---------------------------------------------------------------------------
def test_result_shape_validation():
    with pytest.raises(ValueError):
        SegmentationResult(labels=np.zeros(3), confidences=np.zeros(2),
                           method="x")


def test_result_num_points():
    r = SegmentationResult(labels=np.zeros(5), confidences=np.ones(5), method="x")
    assert r.num_points == 5


# ---------------------------------------------------------------------------
# BaselineSegmenter basics
# ---------------------------------------------------------------------------
def test_baseline_is_prototype():
    seg = BaselineSegmenter(warmup=False)
    assert seg.is_prototype is True


def test_baseline_empty_cloud():
    seg = BaselineSegmenter(warmup=False)
    res = seg.segment(np.empty((0, 3)))
    assert res.num_points == 0


def test_baseline_labels_every_point():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, size=(500, 3))
    res = seg = BaselineSegmenter().segment(pts)
    assert res.num_points == 500
    # every point has a valid class id in [0, 4]
    assert res.labels.min() >= 0 and res.labels.max() <= 4
    assert np.all((res.confidences >= 0) & (res.confidences <= 1))


def test_baseline_detects_flat_ground():
    # A flat ground plane at z=-1.7 should be labeled 'ground'.
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(-20, 20, size=n)
    y = rng.uniform(-20, 20, size=n)
    z = np.full(n, -1.7) + rng.normal(0, 0.01, size=n)
    pts = np.column_stack([x, y, z])
    res = BaselineSegmenter().segment(pts)
    ground_frac = (res.labels == CLASS_ID["ground"]).mean()
    assert ground_frac > 0.9  # almost all flat points are ground


def test_baseline_separates_object_from_ground():
    # Ground + one tall narrow cluster (pedestrian-like) above it.
    rng = np.random.default_rng(2)
    n_g = 3000
    gx = rng.uniform(-15, 15, size=n_g)
    gy = rng.uniform(-15, 15, size=n_g)
    gz = np.full(n_g, -1.7) + rng.normal(0, 0.01, size=n_g)
    ground = np.column_stack([gx, gy, gz])

    n_p = 300
    px = 5 + rng.uniform(-0.2, 0.2, size=n_p)
    py = 5 + rng.uniform(-0.2, 0.2, size=n_p)
    pz = rng.uniform(-1.6, 0.2, size=n_p)  # ~1.8 m tall column
    ped = np.column_stack([px, py, pz])

    pts = np.vstack([ground, ped])
    res = BaselineSegmenter().segment(pts)

    # The pedestrian cluster points should NOT be labeled ground.
    ped_labels = res.labels[n_g:]
    non_ground_frac = (ped_labels != CLASS_ID["ground"]).mean()
    assert non_ground_frac > 0.5


def test_baseline_tall_pole_is_static_obstacle():
    # A tall (3.5 m) narrow pole must be static_obstacle, NOT pedestrian.
    rng = np.random.default_rng(5)
    n_g = 2000
    gx = rng.uniform(-15, 15, size=n_g)
    gy = rng.uniform(-15, 15, size=n_g)
    gz = np.full(n_g, -1.7) + rng.normal(0, 0.01, size=n_g)
    ground = np.column_stack([gx, gy, gz])

    n_p = 800
    px = 10 + rng.uniform(-0.4, 0.4, size=n_p)
    py = 10 + rng.uniform(-0.4, 0.4, size=n_p)
    pz = rng.uniform(-1.5, 2.0, size=n_p)  # 3.5 m tall
    pole = np.column_stack([px, py, pz])

    res = BaselineSegmenter().segment(np.vstack([ground, pole]))
    pole_labels = res.labels[n_g:]
    # majority of pole points should be static_obstacle
    frac_static = (pole_labels == CLASS_ID["static_obstacle"]).mean()
    assert frac_static > 0.5
    # and essentially none should be pedestrian
    frac_ped = (pole_labels == CLASS_ID["pedestrian"]).mean()
    assert frac_ped < 0.1


def test_cluster_downsampling_preserves_object_class():
    # A car-sized cluster classified the same with and without voxel clustering.
    from src.segmentation.baseline_segmenter import BaselineConfig

    rng = np.random.default_rng(7)
    n_g = 4000
    gx = rng.uniform(-20, 20, n_g); gy = rng.uniform(-20, 20, n_g)
    gz = np.full(n_g, -1.7) + rng.normal(0, 0.01, n_g)
    ground = np.column_stack([gx, gy, gz])
    # car: ~4 m x 1.8 m footprint, ~1.5 m tall
    n_c = 1500
    cx = 8 + rng.uniform(-2, 2, n_c); cy = 3 + rng.uniform(-0.9, 0.9, n_c)
    cz = rng.uniform(-1.5, 0.0, n_c)
    car = np.column_stack([cx, cy, cz])
    pts = np.vstack([ground, car])

    slow = BaselineSegmenter(BaselineConfig(cluster_voxel_size=0.0)).segment(pts)
    fast = BaselineSegmenter(BaselineConfig(cluster_voxel_size=0.20)).segment(pts)

    # The car region should be non-ground under both, with the same majority
    # class (vehicle).
    def majority(labels):
        car_labels = labels[n_g:]
        vals, counts = np.unique(car_labels, return_counts=True)
        return vals[np.argmax(counts)]

    assert majority(slow.labels) == majority(fast.labels)
    assert majority(fast.labels) == CLASS_ID["vehicle"]


def test_cluster_voxel_disabled_still_works():
    from src.segmentation.baseline_segmenter import BaselineConfig

    rng = np.random.default_rng(8)
    pts = rng.uniform(-10, 10, size=(600, 3))
    res = BaselineSegmenter(BaselineConfig(cluster_voxel_size=0.0)).segment(pts)
    assert res.num_points == 600
    assert res.labels.min() >= 0 and res.labels.max() <= 4


# ---------------------------------------------------------------------------
# Factory + pretrained stub
# ---------------------------------------------------------------------------
def test_factory_returns_baseline_by_default():
    cfg = {"segmentation": {"method": "baseline"}}
    seg = build_segmenter(cfg)
    assert isinstance(seg, BaselineSegmenter)


def test_factory_unknown_method_raises():
    with pytest.raises(ValueError):
        build_segmenter({"segmentation": {"method": "magic"}})


def test_pretrained_stub_raises_without_weights():
    seg = PretrainedSegmenter(weights_path=None)
    with pytest.raises(NotImplementedError):
        seg.segment(np.zeros((10, 3)))


def test_pretrained_stub_raises_with_missing_weights():
    seg = PretrainedSegmenter(weights_path="models/does_not_exist.pth")
    with pytest.raises(NotImplementedError):
        seg.segment(np.zeros((10, 3)))
