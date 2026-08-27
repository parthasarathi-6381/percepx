"""CLI entry point:  python -m src.benchmarking.benchmark --input <file|dir>

(Also usable as  python -m src.benchmarking  via this module.)
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

# allow running as a module from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.benchmark import (  # noqa: E402
    aggregate,
    benchmark_frame,
    format_report,
)
from src.config import configure_logging, load_config, resolve_path  # noqa: E402
from src.mapping import ResolutionPolicy  # noqa: E402
from src.preprocessing import LidarProcessor, PreprocessConfig  # noqa: E402
from src.segmentation import build_segmenter  # noqa: E402

logger = logging.getLogger("benchmark")


def _find_frames(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(Path(p) for p in glob.glob(str(path / "*.bin")))
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="A .bin frame OR a directory of .bin frames")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate + benchmark a synthetic frame (no dataset needed)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Write full results as JSON to this path")
    ap.add_argument("--no-segment", action="store_true",
                    help="Skip segmentation (benchmark geometry only, faster)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Resolve frames.
    if args.synthetic or not args.input:
        from src.utils.synthetic import write_frame

        f = resolve_path("data/raw/synthetic_000000.bin")
        write_frame(str(f), seed=42)
        frames = [f]
        if not args.synthetic:
            print("No --input given; benchmarking a synthetic frame. "
                  "Use --input <file|dir> for real data.")
    else:
        frames = _find_frames(resolve_path(args.input))

    if not frames:
        print(f"No .bin frames found at {args.input}")
        return 1

    pre = LidarProcessor(PreprocessConfig.from_dict(cfg.get("preprocessing", {})))
    policy = ResolutionPolicy.from_config(cfg)
    segmenter = None if args.no_segment else build_segmenter(cfg)

    results = []
    for fp in frames:
        pc = LidarProcessor.load(fp)
        clean, _ = pre.preprocess(pc)
        seg = None
        if segmenter is not None and clean.size > 0:
            seg = segmenter.segment(clean.points)
        res = benchmark_frame(clean.points, cfg, seg=seg, policy=policy)
        results.append(res)
        print(f"\nFrame: {fp.name}")
        print(format_report(res))

    if len(results) > 1:
        agg = aggregate(results)
        print("\n=== AGGREGATE over", len(results), "frames ===")
        for k, v in agg.items():
            print(f"  {k}: {v}")

    if args.json_out:
        out = {
            "frames": [r.as_dict() for r in results],
            "aggregate": aggregate(results) if len(results) > 1 else None,
        }
        outp = resolve_path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\n[json] wrote -> {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
