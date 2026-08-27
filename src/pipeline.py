"""Central pipeline: LiDAR file -> maps + benchmark, with per-stage timing.

    load -> preprocess -> segment -> uniform 2.5D map -> foveated 2.5D map
         -> benchmark -> (optional) visualization

Exposes timing for every stage:
    load_time, preprocessing_time, segmentation_time,
    uniform_mapping_time, foveated_mapping_time, total_time

Usage
-----
    python -m src.pipeline --input data/raw/000000.bin
    python -m src.pipeline --synthetic
    python -m src.pipeline --input data/raw/000000.bin --viz --json outputs/run.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# allow running as a module from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.benchmark import BenchmarkResult, benchmark_frame  # noqa: E402
from src.config import configure_logging, load_config, resolve_path  # noqa: E402
from src.mapping import (  # noqa: E402
    FoveatedGrid,
    ResolutionPolicy,
    UniformGrid,
    build_foveated_map,
    build_uniform_map,
    map_config,
)
from src.preprocessing import LidarProcessor, PreprocessConfig  # noqa: E402
from src.preprocessing.lidar_processor import PreprocessStats  # noqa: E402
from src.segmentation import build_segmenter  # noqa: E402
from src.segmentation.base_segmenter import SegmentationResult  # noqa: E402
from src.segmentation.classes import NUM_CLASSES  # noqa: E402

logger = logging.getLogger("pipeline")


@dataclass
class PipelineTimings:
    """Per-stage wall-clock timings (seconds)."""

    load_time: float = 0.0
    preprocessing_time: float = 0.0
    segmentation_time: float = 0.0
    uniform_mapping_time: float = 0.0
    foveated_mapping_time: float = 0.0
    total_time: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "load_time": self.load_time,
            "preprocessing_time": self.preprocessing_time,
            "segmentation_time": self.segmentation_time,
            "uniform_mapping_time": self.uniform_mapping_time,
            "foveated_mapping_time": self.foveated_mapping_time,
            "total_time": self.total_time,
        }


@dataclass
class PipelineResult:
    """Everything one pipeline run produces (dashboard-consumable)."""

    input_path: str
    raw_points: np.ndarray
    clean_points: np.ndarray
    preprocess_stats: PreprocessStats
    segmentation: Optional[SegmentationResult]
    uniform_grid: UniformGrid
    foveated_grid: FoveatedGrid
    benchmark: BenchmarkResult
    timings: PipelineTimings
    is_synthetic: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary (no big arrays)."""
        return {
            "input_path": self.input_path,
            "is_synthetic": self.is_synthetic,
            "raw_points": int(self.raw_points.shape[0]),
            "clean_points": int(self.clean_points.shape[0]),
            "preprocess_stats": self.preprocess_stats.as_dict(),
            "segmentation_method": (
                self.segmentation.method if self.segmentation else None
            ),
            "segmentation_is_prototype": (
                self.segmentation.is_prototype if self.segmentation else None
            ),
            "timings": self.timings.as_dict(),
            "benchmark": self.benchmark.as_dict(),
        }


class Pipeline:
    """Orchestrates the full mapping pipeline for a single frame."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.pre = LidarProcessor(PreprocessConfig.from_dict(cfg.get("preprocessing", {})))
        self.policy = ResolutionPolicy.from_config(cfg)
        self.agg = map_config(cfg)
        self.segmenter = build_segmenter(cfg)

    # ------------------------------------------------------------------
    def run(
        self,
        input_path: str | Path,
        do_segment: bool = True,
        is_synthetic: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline on one .bin frame."""
        t_total = time.perf_counter()

        # --- load ---
        t = time.perf_counter()
        pc = LidarProcessor.load(input_path)
        load_time = time.perf_counter() - t
        raw_points = pc.points.copy()

        # --- preprocess ---
        t = time.perf_counter()
        clean, stats = self.pre.preprocess(pc)
        preprocessing_time = time.perf_counter() - t

        # --- segment ---
        seg: Optional[SegmentationResult] = None
        segmentation_time = 0.0
        if do_segment and clean.size > 0:
            t = time.perf_counter()
            try:
                seg = self.segmenter.segment(clean.points)
            except NotImplementedError as exc:
                logger.warning("Segmenter unavailable (%s); continuing unlabeled.", exc)
                seg = None
            segmentation_time = time.perf_counter() - t

        # --- uniform mapping ---
        uni_res = float(self.cfg.get("resolution", {}).get("near", 0.05))
        t = time.perf_counter()
        uniform_grid = build_uniform_map(
            clean.points, seg, resolution=uni_res,
            height_aggregation=self.agg["height_aggregation"],
            confidence_aggregation=self.agg["confidence_aggregation"],
        )
        uniform_mapping_time = time.perf_counter() - t

        # --- foveated mapping ---
        t = time.perf_counter()
        foveated_grid = build_foveated_map(
            clean.points, self.policy, seg,
            height_aggregation=self.agg["height_aggregation"],
            confidence_aggregation=self.agg["confidence_aggregation"],
        )
        foveated_mapping_time = time.perf_counter() - t

        # --- benchmark (reuses the already-classified points) ---
        benchmark = benchmark_frame(clean.points, self.cfg, seg=seg, policy=self.policy)

        total_time = time.perf_counter() - t_total

        timings = PipelineTimings(
            load_time=load_time,
            preprocessing_time=preprocessing_time,
            segmentation_time=segmentation_time,
            uniform_mapping_time=uniform_mapping_time,
            foveated_mapping_time=foveated_mapping_time,
            total_time=total_time,
        )
        logger.info(
            "Pipeline done: load=%.1fms pre=%.1fms seg=%.1fms uni=%.1fms "
            "fov=%.1fms total=%.1fms",
            load_time * 1e3, preprocessing_time * 1e3, segmentation_time * 1e3,
            uniform_mapping_time * 1e3, foveated_mapping_time * 1e3, total_time * 1e3,
        )

        return PipelineResult(
            input_path=str(input_path),
            raw_points=raw_points,
            clean_points=clean.points,
            preprocess_stats=stats,
            segmentation=seg,
            uniform_grid=uniform_grid,
            foveated_grid=foveated_grid,
            benchmark=benchmark,
            timings=timings,
            is_synthetic=is_synthetic,
        )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _resolve_input(args, cfg) -> tuple[Path, bool]:
    """Return (path, is_synthetic)."""
    if args.synthetic:
        from src.utils.synthetic import write_frame

        f = resolve_path("data/raw/synthetic_000000.bin")
        write_frame(str(f), seed=42)
        return f, True
    if args.input:
        return resolve_path(args.input), False
    # default frame, else synthetic fallback
    default = resolve_path(
        Path(cfg["dataset"]["input_path"]) / f"{cfg['dataset']['frame_id']}.bin"
    )
    if default.is_file():
        return default, False
    from src.utils.synthetic import write_frame

    f = resolve_path("data/raw/synthetic_000000.bin")
    write_frame(str(f), seed=42)
    print("No --input and no default frame; using a synthetic frame.")
    return f, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="Path to a KITTI .bin frame")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate + use a synthetic frame (no dataset needed)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-segment", action="store_true")
    ap.add_argument("--viz", action="store_true",
                    help="Write interactive HTML visualizations to outputs/")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Write a JSON summary to this path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    input_path, is_syn = _resolve_input(args, cfg)
    pipeline = Pipeline(cfg)
    result = pipeline.run(input_path, do_segment=not args.no_segment, is_synthetic=is_syn)

    # --- console report ---
    from src.benchmarking.benchmark import format_report

    print(f"\nInput: {result.input_path}"
          + ("  [SYNTHETIC]" if is_syn else ""))
    print("\n--- per-stage timing ---")
    for k, v in result.timings.as_dict().items():
        print(f"  {k:<22}: {v*1e3:8.2f} ms")
    print()
    print(format_report(result.benchmark))

    if result.segmentation is not None and result.segmentation.is_prototype:
        print("\nNOTE: semantic labels are from a PROTOTYPE geometric classifier, "
              "not a trained neural network.")

    # --- optional visualization ---
    if args.viz:
        _write_visualizations(result, cfg)

    # --- optional JSON ---
    if args.json_out:
        outp = resolve_path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            json.dump(result.summary_dict(), fh, indent=2, default=str)
        print(f"\n[json] wrote summary -> {outp}")

    return 0


def _write_visualizations(result: PipelineResult, cfg: Dict[str, Any]) -> None:
    try:
        from src.visualization import plots
    except ImportError as exc:  # pragma: no cover
        print(f"[viz] plotly unavailable ({exc}); skipping.")
        return

    out_dir = resolve_path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_cfg = cfg.get("visualization", {})
    max_pts = int(viz_cfg.get("max_points_render", 120_000))

    figs = {}
    figs["raw_pointcloud"] = plots.plot_raw_pointcloud(result.clean_points, max_pts)
    if result.segmentation is not None:
        figs["semantic_pointcloud"] = plots.plot_semantic_pointcloud(
            result.clean_points, result.segmentation.labels, cfg, max_pts)
    figs["foveated_map"] = plots.plot_grid_map(
        result.foveated_grid.to_arrays(), cfg, title="Foveated 2.5D Map")
    figs["uniform_map"] = plots.plot_grid_map(
        result.uniform_grid.to_arrays(), cfg, title="Uniform 2.5D Map")
    figs["comparison"] = plots.plot_comparison_bars(result.benchmark)

    for name, fig in figs.items():
        p = out_dir / f"{name}.html"
        fig.write_html(str(p))
        print(f"[viz] wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
