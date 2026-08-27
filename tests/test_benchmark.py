"""Tests for the benchmark module (metrics correctness, not absolute speed)."""
from __future__ import annotations

import numpy as np

from src.benchmarking.benchmark import aggregate, benchmark_frame, format_report
from src.mapping import ResolutionPolicy
from src.segmentation import BaselineSegmenter
from src.utils.synthetic import generate_frame


def _cfg():
    return {
        "mapping": {"near_range": 10, "mid_range": 30, "far_range": 60,
                    "max_range": 100, "height_aggregation": "max",
                    "confidence_aggregation": "mean"},
        "resolution": {"near": 0.05, "mid": 0.15, "far": 0.30, "very_far": 0.50},
        "benchmarking": {"repeat": 1, "bytes_per_cell_estimate": 64},
    }


def test_benchmark_basic_metrics():
    pts = generate_frame(seed=11)
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    pts = pts[r <= 100.0]
    res = benchmark_frame(pts, _cfg())

    assert res.num_points == pts.shape[0]
    # foveated must not have MORE cells than uniform
    assert res.foveated.num_cells <= res.uniform.num_cells
    # cell reduction is non-negative and consistent with the counts
    expected = 100 * (res.uniform.num_cells - res.foveated.num_cells) / res.uniform.num_cells
    assert abs(res.cell_reduction_percent - round(expected, 2)) < 0.01
    # logical memory scales with cells, so reduction matches cell reduction
    assert abs(res.logical_memory_reduction_percent - res.cell_reduction_percent) < 0.01


def test_benchmark_fps_positive():
    pts = generate_frame(seed=12)[:, :3]
    res = benchmark_frame(pts, _cfg())
    assert res.uniform.fps > 0
    assert res.foveated.fps > 0
    assert res.fps_improvement > 0


def test_benchmark_with_segmentation():
    pts = generate_frame(seed=13)
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    pts = pts[r <= 100.0]
    seg = BaselineSegmenter().segment(pts)
    res = benchmark_frame(pts, _cfg(), seg=seg)
    # per-zone stats present
    assert set(res.per_zone.keys()) == {"near", "mid", "far", "very_far"}
    # report renders without error
    text = format_report(res)
    assert "BENCHMARK" in text
    assert "cell reduction" in text


def test_aggregate_multi_frame():
    cfg = _cfg()
    results = [benchmark_frame(generate_frame(seed=s)[:, :3], cfg)
               for s in (1, 2, 3)]
    agg = aggregate(results)
    assert agg["frames"] == 3
    assert agg["mean_uniform_cells"] >= agg["mean_foveated_cells"]


def test_benchmark_empty_points():
    res = benchmark_frame(np.empty((0, 3)), _cfg())
    assert res.num_points == 0
    assert res.uniform.num_cells == 0
    assert res.foveated.num_cells == 0
    # no division-by-zero blowups
    assert res.cell_reduction_percent == 0.0
