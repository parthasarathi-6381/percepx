"""Phase 2 / 3 smoke script: load ONE KITTI .bin frame and visualize it.

Usage
-----
    # Use a real KITTI frame:
    python -m scripts.load_and_visualize --input data/raw/000000.bin

    # No dataset yet? Generate a deterministic synthetic frame and use it:
    python -m scripts.load_and_visualize --synthetic

Behavior
--------
1. Loads the frame (KITTI x,y,z,intensity float32).
2. Prints basic stats (count, bounds, radial range).
3. Runs preprocessing and prints the discard report.
4. Writes an interactive HTML view of the raw cloud to outputs/.
   (Falls back to a text summary if plotly is not installed.)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make `src` importable when run as a script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import configure_logging, load_config, resolve_path  # noqa: E402
from src.preprocessing import LidarProcessor, PreprocessConfig  # noqa: E402

logger = logging.getLogger("load_and_visualize")


def _print_cloud_stats(name: str, pts: np.ndarray) -> None:
    if pts.shape[0] == 0:
        print(f"  [{name}] EMPTY")
        return
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    print(f"  [{name}] points = {pts.shape[0]:,}")
    print(
        f"      x: [{pts[:,0].min():8.2f}, {pts[:,0].max():8.2f}]   "
        f"y: [{pts[:,1].min():8.2f}, {pts[:,1].max():8.2f}]   "
        f"z: [{pts[:,2].min():8.2f}, {pts[:,2].max():8.2f}]"
    )
    print(f"      radial r: [{r.min():.2f}, {r.max():.2f}] m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="Path to a KITTI .bin frame")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate + use a deterministic synthetic frame (no dataset needed)",
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip writing the HTML visualization",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    # --- Resolve the input frame ---
    if args.synthetic:
        from src.utils.synthetic import write_frame

        out_path = resolve_path("data/raw/synthetic_000000.bin")
        write_frame(str(out_path), seed=42)
        input_path = out_path
        print(f"[synthetic] generated deterministic frame -> {input_path}")
    elif args.input:
        input_path = resolve_path(args.input)
    else:
        # Try the configured default; if missing, fall back to synthetic.
        default = resolve_path(
            Path(cfg["dataset"]["input_path"]) / f"{cfg['dataset']['frame_id']}.bin"
        )
        if default.is_file():
            input_path = default
        else:
            print(
                f"No --input given and default frame not found ({default}).\n"
                "Falling back to a synthetic frame. Use --input <file.bin> for real data."
            )
            from src.utils.synthetic import write_frame

            input_path = resolve_path("data/raw/synthetic_000000.bin")
            write_frame(str(input_path), seed=42)

    # --- Phase 2: load ---
    print(f"\n=== Loading: {input_path} ===")
    pc = LidarProcessor.load(input_path)
    _print_cloud_stats("raw", pc.points)

    # --- Phase 3: preprocess ---
    pre_cfg = PreprocessConfig.from_dict(cfg.get("preprocessing", {}))
    processor = LidarProcessor(pre_cfg)
    clean, stats = processor.preprocess(pc)

    print("\n=== Preprocessing report ===")
    for k, v in stats.as_dict().items():
        print(f"  {k}: {v}")
    _print_cloud_stats("clean", clean.points)

    # --- Visualization ---
    if not args.no_viz and clean.size > 0:
        try:
            from src.visualization.plots import plot_raw_pointcloud

            viz_cfg = cfg.get("visualization", {})
            fig = plot_raw_pointcloud(
                clean.points,
                max_points=int(viz_cfg.get("max_points_render", 120_000)),
                point_size=float(viz_cfg.get("point_size", 1.5)),
            )
            out_dir = resolve_path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_html = out_dir / "raw_pointcloud.html"
            fig.write_html(str(out_html))
            print(f"\n[viz] wrote interactive view -> {out_html}")
        except ImportError as exc:
            print(f"\n[viz] plotly not available ({exc}); skipping HTML view.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
