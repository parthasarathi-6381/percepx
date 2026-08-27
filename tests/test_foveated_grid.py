"""Unit tests for the FoveatedGrid: zone routing, boundary handling, no loss."""
from __future__ import annotations

import numpy as np
import pytest

from src.mapping.foveated_grid import FoveatedGrid
from src.mapping.resolution_policy import ResolutionPolicy


@pytest.fixture
def policy():
    return ResolutionPolicy(
        near_range=10.0, mid_range=30.0, far_range=60.0, max_range=100.0,
        res_near=0.05, res_mid=0.15, res_far=0.30, res_very_far=0.50,
    )


def _point_at_radius(r, z=0.0, angle=0.0):
    """A single point at horizontal distance r along `angle`."""
    return np.array([[r * np.cos(angle), r * np.sin(angle), z, 0.1]])


# ---------------------------------------------------------------------------
# Zone routing: each point lands in exactly the right zone grid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "r,zone",
    [
        (5.0, "near"), (10.0, "near"),
        (20.0, "mid"), (30.0, "mid"),
        (45.0, "far"), (60.0, "far"),
        (80.0, "very_far"), (100.0, "very_far"),
    ],
)
def test_point_routed_to_correct_zone(policy, r, zone):
    g = FoveatedGrid(policy)
    g.insert(_point_at_radius(r)[:, :3])
    counts = g.num_cells_by_zone()
    assert counts[zone] == 1
    # all other zones empty
    assert sum(v for k, v in counts.items() if k != zone) == 0


def test_out_of_range_discarded_and_counted(policy):
    g = FoveatedGrid(policy)
    pts = np.vstack([
        _point_at_radius(5.0)[:, :3],     # near
        _point_at_radius(150.0)[:, :3],   # discard
        _point_at_radius(200.0)[:, :3],   # discard
    ])
    g.insert(pts)
    assert g.discarded_out_of_range == 2
    assert g.num_cells() == 1  # only the near point made a cell


def test_negative_coordinates_work(policy):
    # Point at (-30, -40) -> r = 50 -> "far" zone.
    g = FoveatedGrid(policy)
    g.insert(np.array([[-30.0, -40.0, 1.0]]))
    counts = g.num_cells_by_zone()
    assert counts["far"] == 1


# ---------------------------------------------------------------------------
# No point loss: every retained point is aggregated exactly once
# ---------------------------------------------------------------------------
def test_no_point_loss_conservation(policy):
    rng = np.random.default_rng(3)
    n = 5000
    r = rng.uniform(0, 120, size=n)          # some beyond 100 m
    th = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th), rng.normal(size=n)])

    g = FoveatedGrid(policy)
    g.insert(pts)
    stats = g.stats()

    # points routed to zones + discarded == n
    routed = sum(z["points"] for z in stats["by_zone"].values())
    assert routed + stats["discarded_out_of_range"] == n

    # sum of point_count over all cells == routed (nothing dropped in aggregation)
    total_in_cells = 0
    for name, grid in g.zone_grids.items():
        if grid.num_cells():
            total_in_cells += int(grid.to_arrays()["point_count"].sum())
    assert total_in_cells == routed


# ---------------------------------------------------------------------------
# Cell resolution matches the zone
# ---------------------------------------------------------------------------
def test_cell_resolution_matches_zone(policy):
    g = FoveatedGrid(policy)
    g.insert(np.vstack([
        _point_at_radius(5.0)[:, :3],
        _point_at_radius(20.0)[:, :3],
        _point_at_radius(45.0)[:, :3],
        _point_at_radius(80.0)[:, :3],
    ]))
    arr = g.to_arrays()
    res_by_zone = {}
    for res, zi in zip(arr["resolution"], arr["zone_index"]):
        res_by_zone[int(zi)] = res
    assert res_by_zone[0] == 0.05
    assert res_by_zone[1] == 0.15
    assert res_by_zone[2] == 0.30
    assert res_by_zone[3] == 0.50


# ---------------------------------------------------------------------------
# Foveated produces fewer cells than uniform (the whole point)
# ---------------------------------------------------------------------------
def test_foveated_fewer_cells_than_uniform(policy):
    from src.mapping.uniform_grid import UniformGrid
    from src.utils.synthetic import generate_frame

    pts = generate_frame(seed=1)[:, :3]
    # keep within range
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    pts = pts[r <= 100.0]

    uni = UniformGrid(resolution=0.05)
    uni.insert(pts)
    fov = FoveatedGrid(policy)
    fov.insert(pts)

    assert fov.num_cells() < uni.num_cells()


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------
def test_empty_insert(policy):
    g = FoveatedGrid(policy)
    g.insert(np.empty((0, 3)))
    assert g.num_cells() == 0
    assert g.to_arrays()["x"].shape == (0,)
