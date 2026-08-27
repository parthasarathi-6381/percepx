"""Phase 10: explicit edge cases from the spec + real-data-format readiness."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.mapping import FoveatedGrid, ResolutionPolicy, UniformGrid
from src.pipeline import Pipeline
from src.preprocessing import LidarProcessor, PointCloud, PreprocessConfig


# ---------------------------------------------------------------------------
# Real KITTI .bin format readiness: a raw float32 x,y,z,intensity buffer that
# we did NOT create via our synthetic helper must flow through the pipeline.
# ---------------------------------------------------------------------------
def test_real_kitti_format_flows_through_pipeline(tmp_path):
    # Hand-build a KITTI-style buffer (this is exactly the on-disk layout KITTI
    # uses) with a ground plane + one object, save as .bin, run the pipeline.
    rng = np.random.default_rng(0)
    n = 8000
    x = rng.uniform(-40, 40, n).astype(np.float32)
    y = rng.uniform(-40, 40, n).astype(np.float32)
    z = np.full(n, -1.7, dtype=np.float32) + rng.normal(0, 0.02, n).astype(np.float32)
    inten = rng.uniform(0, 1, n).astype(np.float32)
    buf = np.column_stack([x, y, z, inten]).astype(np.float32)

    f = tmp_path / "000000.bin"
    buf.tofile(str(f))  # raw KITTI layout, NOT via our writer

    cfg = load_config()
    result = Pipeline(cfg).run(str(f))
    assert result.clean_points.shape[0] > 0
    assert result.uniform_grid.num_cells() > 0
    assert result.foveated_grid.num_cells() > 0


# ---------------------------------------------------------------------------
# Point exactly at each zone boundary -> lands in the correct foveated cell
# at the correct resolution.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "r,expected_res",
    [(10.0, 0.05), (30.0, 0.15), (60.0, 0.30), (100.0, 0.50)],
)
def test_exact_boundary_point_gets_correct_resolution(r, expected_res):
    policy = ResolutionPolicy(10, 30, 60, 100, 0.05, 0.15, 0.30, 0.50)
    g = FoveatedGrid(policy)
    # place exactly along +x at distance r
    g.insert(np.array([[r, 0.0, 0.5]]))
    arr = g.to_arrays()
    assert arr["x"].shape[0] == 1
    assert arr["resolution"][0] == expected_res


def test_point_just_outside_max_range_discarded():
    policy = ResolutionPolicy(10, 30, 60, 100, 0.05, 0.15, 0.30, 0.50)
    g = FoveatedGrid(policy)
    g.insert(np.array([[100.001, 0.0, 0.0]]))
    assert g.num_cells() == 0
    assert g.discarded_out_of_range == 1


# ---------------------------------------------------------------------------
# Duplicate points at the exact same location -> one cell, correct count.
# ---------------------------------------------------------------------------
def test_exact_duplicate_points():
    pts = np.tile(np.array([[3.3, 3.3, 1.0]]), (25, 1))
    g = UniformGrid(resolution=0.05)
    g.insert(pts)
    assert g.num_cells() == 1
    assert g.to_arrays()["point_count"][0] == 25


# ---------------------------------------------------------------------------
# Invalid values are removed before mapping (integration through preprocess).
# ---------------------------------------------------------------------------
def test_invalid_values_removed_before_mapping():
    pts = np.array([
        [1.0, 1.0, 0.0, 0.1],
        [np.nan, 1.0, 0.0, 0.1],
        [1.0, np.inf, 0.0, 0.1],
        [2.0, 2.0, 0.0, 0.1],
    ], dtype=np.float32)
    proc = LidarProcessor(PreprocessConfig(max_range=100, z_min=-3, z_max=3))
    clean, stats = proc.preprocess(PointCloud(pts))
    assert stats.removed_invalid == 2
    assert np.isfinite(clean.points).all()


# ---------------------------------------------------------------------------
# Empty everything: pipeline on an empty frame must not crash.
# ---------------------------------------------------------------------------
def test_pipeline_on_empty_frame(tmp_path):
    f = tmp_path / "empty.bin"
    np.array([], dtype=np.float32).tofile(str(f))
    cfg = load_config()
    result = Pipeline(cfg).run(str(f))
    assert result.clean_points.shape[0] == 0
    assert result.uniform_grid.num_cells() == 0
    assert result.foveated_grid.num_cells() == 0
    assert result.benchmark.cell_reduction_percent == 0.0


# ---------------------------------------------------------------------------
# Negative-only quadrant: all points at negative x AND y must map correctly.
# ---------------------------------------------------------------------------
def test_all_negative_quadrant():
    rng = np.random.default_rng(1)
    x = rng.uniform(-30, -1, 500)
    y = rng.uniform(-30, -1, 500)
    z = rng.uniform(-1, 1, 500)
    pts = np.column_stack([x, y, z])
    g = UniformGrid(resolution=0.5)
    g.insert(pts)
    arr = g.to_arrays()
    # all resolved cell origins should be negative
    assert (arr["x"] < 0).all()
    assert (arr["y"] < 0).all()
