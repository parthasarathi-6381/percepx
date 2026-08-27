"""Unit tests for the uniform 2.5D grid."""
from __future__ import annotations

import numpy as np
import pytest

from src.mapping.cell import NO_CLASS
from src.mapping.uniform_grid import UniformGrid


# ---------------------------------------------------------------------------
# Coordinate calculation
# ---------------------------------------------------------------------------
def test_cell_index_basic():
    g = UniformGrid(resolution=0.05)
    gx, gy = g.cell_index(np.array([0.0, 0.049, 0.05, 0.10]),
                          np.array([0.0, 0.0, 0.0, 0.0]))
    # 0.0->0, 0.049->0, 0.05->1, 0.10->2
    assert list(gx) == [0, 0, 1, 2]


def test_cell_index_negative_coords():
    # floor semantics: -0.01 -> cell -1, not 0
    g = UniformGrid(resolution=0.05)
    gx, gy = g.cell_index(np.array([-0.01, -0.05, -0.06]),
                          np.array([-0.01, -0.05, -0.06]))
    assert list(gx) == [-1, -1, -2]
    assert list(gy) == [-1, -1, -2]


def test_invalid_resolution_raises():
    with pytest.raises(ValueError):
        UniformGrid(resolution=0.0)
    with pytest.raises(ValueError):
        UniformGrid(resolution=-1.0)


# ---------------------------------------------------------------------------
# Point-to-cell assignment + counting
# ---------------------------------------------------------------------------
def test_single_point_one_cell():
    g = UniformGrid(resolution=1.0)
    g.insert(np.array([[0.5, 0.5, 2.0]]))
    assert g.num_cells() == 1
    arr = g.to_arrays()
    assert arr["x"][0] == 0.0 and arr["y"][0] == 0.0  # cell origin
    assert arr["point_count"][0] == 1
    assert arr["elevation"][0] == 2.0


def test_duplicate_points_same_cell():
    g = UniformGrid(resolution=1.0)
    pts = np.array([[0.1, 0.1, 1.0], [0.2, 0.2, 5.0], [0.9, 0.9, 3.0]])
    g.insert(pts)
    assert g.num_cells() == 1          # all in cell (0,0)
    arr = g.to_arrays()
    assert arr["point_count"][0] == 3
    assert arr["elevation"][0] == 5.0  # max z by default


def test_points_span_multiple_cells():
    g = UniformGrid(resolution=1.0)
    pts = np.array([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [-0.5, -0.5, 1.0]])
    g.insert(pts)
    assert g.num_cells() == 3


# ---------------------------------------------------------------------------
# Height aggregation modes
# ---------------------------------------------------------------------------
def test_height_aggregation_max_mean_min():
    pts = np.array([[0.1, 0.1, 1.0], [0.2, 0.2, 3.0], [0.3, 0.3, 5.0]])

    gmax = UniformGrid(resolution=1.0, height_aggregation="max")
    gmax.insert(pts)
    assert gmax.to_arrays()["elevation"][0] == 5.0

    gmin = UniformGrid(resolution=1.0, height_aggregation="min")
    gmin.insert(pts)
    assert gmin.to_arrays()["elevation"][0] == 1.0

    gmean = UniformGrid(resolution=1.0, height_aggregation="mean")
    gmean.insert(pts)
    assert gmean.to_arrays()["elevation"][0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Semantic aggregation (multiple classes in one cell)
# ---------------------------------------------------------------------------
def test_semantic_vote_by_confidence():
    # Cell (0,0): class 1 with high confidence should win over class 2 low conf.
    pts = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]])
    labels = np.array([1, 2, 1])
    conf = np.array([0.9, 0.5, 0.8])
    g = UniformGrid(resolution=1.0)
    g.insert(pts, labels=labels, confidences=conf, n_classes=5)
    arr = g.to_arrays()
    assert arr["semantic_class"][0] == 1  # summed conf 1.7 > 0.5


def test_semantic_single_class_wins():
    pts = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0]])
    labels = np.array([3, 3])
    g = UniformGrid(resolution=1.0)
    g.insert(pts, labels=labels, confidences=np.array([0.7, 0.7]), n_classes=5)
    arr = g.to_arrays()
    assert arr["semantic_class"][0] == 3


def test_no_labels_gives_no_class():
    g = UniformGrid(resolution=1.0)
    g.insert(np.array([[0.1, 0.1, 0.0]]))
    assert g.to_arrays()["semantic_class"][0] == NO_CLASS


# ---------------------------------------------------------------------------
# Confidence aggregation
# ---------------------------------------------------------------------------
def test_confidence_mean_and_max():
    pts = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0]])
    labels = np.array([1, 1])
    conf = np.array([0.4, 0.8])

    gmean = UniformGrid(resolution=1.0, confidence_aggregation="mean")
    gmean.insert(pts, labels=labels, confidences=conf, n_classes=5)
    assert gmean.to_arrays()["confidence"][0] == pytest.approx(0.6)

    gmaxc = UniformGrid(resolution=1.0, confidence_aggregation="max")
    gmaxc.insert(pts, labels=labels, confidences=conf, n_classes=5)
    assert gmaxc.to_arrays()["confidence"][0] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------
def test_empty_insert():
    g = UniformGrid(resolution=0.05)
    g.insert(np.empty((0, 3)))
    assert g.num_cells() == 0
    assert g.to_arrays()["x"].shape == (0,)


# ---------------------------------------------------------------------------
# Serializable Cell output
# ---------------------------------------------------------------------------
def test_cells_serializable():
    g = UniformGrid(resolution=1.0)
    g.insert(np.array([[0.5, 0.5, 2.0]]), labels=np.array([1]),
             confidences=np.array([0.9]), n_classes=5)
    cells = g.cells()
    assert len(cells) == 1
    d = cells[0].to_dict()
    assert d["occupancy"] is True
    assert d["point_count"] == 1
    assert d["semantic_class"] == 1
    assert d["zone"] == "uniform"


def test_stats():
    g = UniformGrid(resolution=1.0)
    g.insert(np.array([[0.1, 0.1, 0.0], [5.5, 5.5, 0.0]]))
    s = g.stats()
    assert s["num_cells"] == 2
    assert s["total_points"] == 2
    assert s["resolution"] == 1.0
