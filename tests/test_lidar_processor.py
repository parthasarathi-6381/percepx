"""Unit tests for the LiDAR loader and preprocessing.

Deterministic and self-contained: no external dataset required. Frames are
built in-memory or written to a pytest tmp_path.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import LidarProcessor, PointCloud, PreprocessConfig


# ---------------------------------------------------------------------------
# PointCloud container
# ---------------------------------------------------------------------------
def test_pointcloud_rejects_wrong_shape():
    with pytest.raises(ValueError):
        PointCloud(np.zeros((5, 3)))  # only 3 columns, need 4


def test_pointcloud_column_accessors():
    pts = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.9]], dtype=np.float32)
    pc = PointCloud(pts)
    assert pc.size == 2
    assert np.allclose(pc.x, [1, 4])
    assert np.allclose(pc.y, [2, 5])
    assert np.allclose(pc.z, [3, 6])
    assert np.allclose(pc.intensity, [0.5, 0.9])


def test_radial_distance_matches_manual():
    pts = np.array([[3, 4, 0, 0], [-3, -4, 1, 0]], dtype=np.float32)
    pc = PointCloud(pts)
    r = pc.radial_distance()
    # both are 3-4-5 triangles -> r = 5, and negatives must work
    assert np.allclose(r, [5.0, 5.0])


# ---------------------------------------------------------------------------
# Loading (round-trip via save)
# ---------------------------------------------------------------------------
def test_load_roundtrip(tmp_path):
    pts = np.array(
        [[1, 2, 3, 0.1], [-4, -5, -6, 0.2], [7, 8, 9, 0.3]], dtype=np.float32
    )
    f = tmp_path / "frame.bin"
    LidarProcessor.save(PointCloud(pts), f)
    loaded = LidarProcessor.load(f)
    assert loaded.size == 3
    assert np.allclose(loaded.points, pts)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LidarProcessor.load(tmp_path / "does_not_exist.bin")


def test_load_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    np.array([], dtype=np.float32).tofile(str(f))
    pc = LidarProcessor.load(f)
    assert pc.size == 0
    assert pc.points.shape == (0, 4)


def test_load_bad_float_count_raises(tmp_path):
    f = tmp_path / "bad.bin"
    # 5 floats -> not divisible by 4
    np.array([1, 2, 3, 4, 5], dtype=np.float32).tofile(str(f))
    with pytest.raises(ValueError):
        LidarProcessor.load(f)


# ---------------------------------------------------------------------------
# Preprocessing: invalid points
# ---------------------------------------------------------------------------
def test_remove_invalid_points():
    pts = np.array(
        [
            [1, 1, 0, 0.5],
            [np.nan, 2, 0, 0.5],
            [3, np.inf, 0, 0.5],
            [4, 4, -np.inf, 0.5],
            [5, 5, 0, 0.5],
        ],
        dtype=np.float32,
    )
    proc = LidarProcessor(PreprocessConfig(max_range=100.0, z_min=-10, z_max=10))
    clean, stats = proc.preprocess(PointCloud(pts))
    assert stats.removed_invalid == 3
    assert clean.size == 2
    assert np.isfinite(clean.points).all()


# ---------------------------------------------------------------------------
# Preprocessing: range filtering (the important boundary cases)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "r,expected_kept",
    [
        (0.0, True),     # exactly min
        (50.0, True),    # inside
        (100.0, True),   # exactly max -> kept (inclusive)
        (100.001, False),  # just outside
        (150.0, False),  # far outside
    ],
)
def test_range_filter_boundaries(r, expected_kept):
    # place a single point at horizontal distance r along +x
    pts = np.array([[r, 0.0, 0.0, 0.1]], dtype=np.float32)
    proc = LidarProcessor(PreprocessConfig(min_range=0.0, max_range=100.0,
                                           z_min=-10, z_max=10))
    clean, _ = proc.preprocess(PointCloud(pts))
    assert (clean.size == 1) == expected_kept


def test_range_filter_negative_coords_kept():
    # Point at (-30, -40) -> r = 50, must be kept (negative coords must work).
    pts = np.array([[-30.0, -40.0, 0.0, 0.1]], dtype=np.float32)
    proc = LidarProcessor(PreprocessConfig(max_range=100.0, z_min=-10, z_max=10))
    clean, _ = proc.preprocess(PointCloud(pts))
    assert clean.size == 1


# ---------------------------------------------------------------------------
# Preprocessing: z crop
# ---------------------------------------------------------------------------
def test_z_crop():
    pts = np.array(
        [
            [1, 1, -5.0, 0.1],  # below z_min
            [1, 1, 0.0, 0.1],   # inside
            [1, 1, 5.0, 0.1],   # above z_max
        ],
        dtype=np.float32,
    )
    proc = LidarProcessor(PreprocessConfig(z_min=-3.0, z_max=3.0, max_range=100))
    clean, stats = proc.preprocess(PointCloud(pts))
    assert clean.size == 1
    assert stats.removed_z_crop == 2


# ---------------------------------------------------------------------------
# Preprocessing: empty input
# ---------------------------------------------------------------------------
def test_preprocess_empty_cloud():
    proc = LidarProcessor()
    clean, stats = proc.preprocess(PointCloud(np.empty((0, 4), dtype=np.float32)))
    assert clean.size == 0
    assert stats.input_points == 0
    assert stats.output_points == 0
    assert "empty input cloud" in stats.notes


# ---------------------------------------------------------------------------
# Preprocessing: stats accounting (no point silently discarded)
# ---------------------------------------------------------------------------
def test_stats_conservation():
    # Build a mix: valid, invalid, out-of-range, out-of-z.
    pts = np.array(
        [
            [1, 1, 0, 0.1],          # valid
            [np.nan, 1, 0, 0.1],     # invalid
            [200, 0, 0, 0.1],        # out of range
            [1, 1, 50, 0.1],         # out of z
            [2, 2, 0, 0.1],          # valid
        ],
        dtype=np.float32,
    )
    proc = LidarProcessor(PreprocessConfig(max_range=100, z_min=-3, z_max=3))
    clean, stats = proc.preprocess(PointCloud(pts))
    # input = output + everything removed
    assert stats.input_points == stats.output_points + stats.total_removed
    assert clean.size == 2


# ---------------------------------------------------------------------------
# Voxel downsampling
# ---------------------------------------------------------------------------
def test_voxel_downsample_merges_duplicates():
    # 10 points in the same tiny region should collapse to ~1 voxel.
    pts = np.tile(np.array([[1.0, 1.0, 0.0, 0.5]], dtype=np.float32), (10, 1))
    cfg = PreprocessConfig(voxel_downsample=True, voxel_size=0.5,
                           max_range=100, z_min=-3, z_max=3)
    proc = LidarProcessor(cfg)
    clean, stats = proc.preprocess(PointCloud(pts))
    assert clean.size == 1
    assert stats.removed_voxel == 9


def test_voxel_downsample_deterministic():
    from src.utils.synthetic import generate_frame

    pts = generate_frame(seed=7)
    cfg = PreprocessConfig(voxel_downsample=True, voxel_size=0.2,
                           max_range=100, z_min=-3, z_max=3)
    proc = LidarProcessor(cfg)
    a, _ = proc.preprocess(PointCloud(pts.copy()))
    b, _ = proc.preprocess(PointCloud(pts.copy()))
    assert a.size == b.size
    assert np.allclose(a.points, b.points)
