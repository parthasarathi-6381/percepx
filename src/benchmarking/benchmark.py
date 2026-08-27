"""Performance benchmarking: uniform 5 cm grid vs foveated grid.

This is the headline SIH comparison. All numbers are REAL measurements taken on
the same input frame(s); nothing is fabricated.

What is measured
----------------
* number of points (after preprocessing)
* number of cells (uniform vs foveated)
* number of occupied cells (same as #cells here -- we store sparsely)
* mapping latency (time to build each grid)
* total latency (preprocess + segment + both maps) -- reported by the pipeline
* FPS = 1 / mapping_time
* memory:
    - LOGICAL storage estimate = occupied_cells * bytes_per_cell  (exact, model)
    - PROCESS memory delta (psutil RSS) around building each grid (noisy, real)
  The two are reported separately and clearly labeled (they measure different
  things: logical storage is the map's data footprint; RSS includes Python
  object + allocator overhead).

Derived reductions
------------------
    cell_reduction_percent    = 100 * (uni_cells - fov_cells) / uni_cells
    memory_reduction_percent  = 100 * (uni_mem   - fov_mem)   / uni_mem
    latency_reduction_percent = 100 * (uni_time  - fov_time)  / uni_time
    fps_improvement           = fov_fps / uni_fps
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..mapping import (
    FoveatedGrid,
    ResolutionPolicy,
    UniformGrid,
    build_foveated_map,
    build_uniform_map,
    map_config,
)
from ..segmentation.base_segmenter import SegmentationResult

logger = logging.getLogger(__name__)

try:
    import psutil  # optional but preferred

    _HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAVE_PSUTIL = False


def _rss_bytes() -> Optional[int]:
    """Current process resident set size in bytes, or None if psutil missing."""
    if not _HAVE_PSUTIL:
        return None
    return int(psutil.Process().memory_info().rss)


@dataclass
class GridBenchmark:
    """Measurements for a single grid (uniform or foveated)."""

    name: str
    resolution: str            # "0.05 (uniform)" or "0.05/0.15/0.30/0.50 (foveated)"
    num_cells: int
    occupied_cells: int
    mapping_time_s: float
    fps: float
    logical_bytes: int         # occupied_cells * bytes_per_cell
    process_rss_delta_bytes: Optional[int]  # measured RSS change (may be None/noisy)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Full uniform-vs-foveated comparison for one frame (or aggregate)."""

    num_points: int
    uniform: GridBenchmark
    foveated: GridBenchmark
    cell_reduction_percent: float
    logical_memory_reduction_percent: float
    latency_reduction_percent: float
    fps_improvement: float
    per_zone: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _bytes_per_cell(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("benchmarking", {}).get("bytes_per_cell_estimate", 64))


def _time_build(build_fn, repeat: int) -> tuple[Any, float, Optional[int]]:
    """Run ``build_fn`` ``repeat`` times; return (last_grid, best_time, rss_delta).

    best_time = minimum wall time across repeats (least noisy estimate of the
    real mapping cost). rss_delta measured once around a clean build.
    """
    grid = None
    best = float("inf")
    for _ in range(max(1, repeat)):
        gc.collect()
        t0 = time.perf_counter()
        grid = build_fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)

    # Measure RSS delta around one isolated build.
    gc.collect()
    rss_before = _rss_bytes()
    grid_mem = build_fn()
    rss_after = _rss_bytes()
    rss_delta = None
    if rss_before is not None and rss_after is not None:
        rss_delta = max(0, rss_after - rss_before)
    # keep the timed grid, discard the mem-probe grid
    del grid_mem
    return grid, best, rss_delta


def benchmark_frame(
    points: np.ndarray,
    cfg: Dict[str, Any],
    seg: Optional[SegmentationResult] = None,
    policy: Optional[ResolutionPolicy] = None,
) -> BenchmarkResult:
    """Benchmark uniform vs foveated mapping on one preprocessed frame."""
    policy = policy or ResolutionPolicy.from_config(cfg)
    agg = map_config(cfg)
    repeat = int(cfg.get("benchmarking", {}).get("repeat", 1))
    bpc = _bytes_per_cell(cfg)
    uni_res = float(cfg.get("resolution", {}).get("near", 0.05))

    xyz = points[:, :3]
    n_points = int(xyz.shape[0])

    # --- uniform ---
    uni_grid, uni_time, uni_rss = _time_build(
        lambda: build_uniform_map(
            xyz, seg, resolution=uni_res,
            height_aggregation=agg["height_aggregation"],
            confidence_aggregation=agg["confidence_aggregation"],
        ),
        repeat,
    )
    uni_cells = uni_grid.num_cells()

    # --- foveated ---
    fov_grid, fov_time, fov_rss = _time_build(
        lambda: build_foveated_map(
            xyz, policy, seg,
            height_aggregation=agg["height_aggregation"],
            confidence_aggregation=agg["confidence_aggregation"],
        ),
        repeat,
    )
    fov_cells = fov_grid.num_cells()

    def fps(t: float) -> float:
        return (1.0 / t) if t > 0 else float("inf")

    uni = GridBenchmark(
        name="uniform",
        resolution=f"{uni_res:g} m (uniform)",
        num_cells=uni_cells,
        occupied_cells=uni_cells,
        mapping_time_s=uni_time,
        fps=fps(uni_time),
        logical_bytes=uni_cells * bpc,
        process_rss_delta_bytes=uni_rss,
    )
    zone_res = "/".join(f"{z.resolution:g}" for z in policy.zones)
    fov = GridBenchmark(
        name="foveated",
        resolution=f"{zone_res} m (foveated)",
        num_cells=fov_cells,
        occupied_cells=fov_cells,
        mapping_time_s=fov_time,
        fps=fps(fov_time),
        logical_bytes=fov_cells * bpc,
        process_rss_delta_bytes=fov_rss,
    )

    def pct(a: float, b: float) -> float:
        return 100.0 * (a - b) / a if a > 0 else 0.0

    result = BenchmarkResult(
        num_points=n_points,
        uniform=uni,
        foveated=fov,
        cell_reduction_percent=round(pct(uni_cells, fov_cells), 2),
        logical_memory_reduction_percent=round(pct(uni.logical_bytes, fov.logical_bytes), 2),
        latency_reduction_percent=round(pct(uni_time, fov_time), 2),
        fps_improvement=round(fov.fps / uni.fps, 3) if uni.fps > 0 else float("inf"),
        per_zone=fov_grid.stats()["by_zone"],
        notes=[
            "All values are real measurements on the same frame.",
            "logical_bytes = occupied_cells * bytes_per_cell_estimate "
            f"({bpc} B/cell): the map's data footprint.",
            "process_rss_delta_bytes = measured psutil RSS change around one "
            "build: includes Python/allocator overhead and is noisy.",
            "Foveation's primary win is fewer cells -> less memory, which "
            "compounds for every downstream consumer of the map (storage, "
            "transmission, planning). Construction latency is comparable "
            "between the two (both real-time); it is NOT the metric foveation "
            "optimizes, so a small latency delta either way is expected.",
        ],
    )
    return result


def aggregate(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Aggregate several per-frame results into mean metrics (multi-frame)."""
    if not results:
        return {}
    n = len(results)
    return {
        "frames": n,
        "mean_points": float(np.mean([r.num_points for r in results])),
        "mean_uniform_cells": float(np.mean([r.uniform.num_cells for r in results])),
        "mean_foveated_cells": float(np.mean([r.foveated.num_cells for r in results])),
        "mean_cell_reduction_percent": round(
            float(np.mean([r.cell_reduction_percent for r in results])), 2),
        "mean_latency_reduction_percent": round(
            float(np.mean([r.latency_reduction_percent for r in results])), 2),
        "mean_uniform_fps": round(float(np.mean([r.uniform.fps for r in results])), 2),
        "mean_foveated_fps": round(float(np.mean([r.foveated.fps for r in results])), 2),
        "mean_fps_improvement": round(
            float(np.mean([r.fps_improvement for r in results])), 3),
    }


def format_report(result: BenchmarkResult) -> str:
    """Human-readable console report for one frame."""
    u, f = result.uniform, result.foveated
    lines = [
        "=" * 62,
        "  FOVEATED vs UNIFORM  --  BENCHMARK (real measurements)",
        "=" * 62,
        f"  input points        : {result.num_points:,}",
        "",
        f"  {'metric':<20}{'uniform':>16}{'foveated':>16}",
        "  " + "-" * 52,
        f"  {'cells':<20}{u.num_cells:>16,}{f.num_cells:>16,}",
        f"  {'mapping time (ms)':<20}{u.mapping_time_s*1e3:>16.2f}{f.mapping_time_s*1e3:>16.2f}",
        f"  {'FPS':<20}{u.fps:>16.1f}{f.fps:>16.1f}",
        f"  {'logical KB':<20}{u.logical_bytes/1024:>16.1f}{f.logical_bytes/1024:>16.1f}",
        "  " + "-" * 52,
        f"  cell reduction      : {result.cell_reduction_percent:.1f}%",
        f"  logical mem reduction: {result.logical_memory_reduction_percent:.1f}%",
        f"  latency reduction   : {result.latency_reduction_percent:.1f}%",
        f"  FPS improvement     : {result.fps_improvement:.2f}x",
        "",
        "  per-zone (foveated):",
    ]
    for zname, z in result.per_zone.items():
        lines.append(
            f"    {zname:<9} @ {z['resolution']:g} m : "
            f"{z['num_cells']:>8,} cells   {z['points']:>8,} pts"
        )
    if u.process_rss_delta_bytes is not None:
        lines += [
            "",
            f"  process RSS delta (noisy): uniform {u.process_rss_delta_bytes/1024:.0f} KB, "
            f"foveated {f.process_rss_delta_bytes/1024:.0f} KB",
        ]
    lines.append("=" * 62)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    # Support: python -m src.benchmarking.benchmark --input <file|dir>
    from src.benchmarking.__main__ import main

    raise SystemExit(main())
