"""Render KITTI .bin LiDAR frames as PNG images (BEV, side, 3D, semantic).

Usage:
    python -m scripts.render_frames                 # frame 0, all views
    python -m scripts.render_frames --frame 10
    python -m scripts.render_frames --all-bev       # BEV thumbnail of every frame
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write files, no window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "frame_images"


def load(path: str) -> np.ndarray:
    """KITTI Velodyne .bin -> (N,4) float32 [x,y,z,intensity]."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def bev(pts: np.ndarray, title: str, out: Path, color_by: str = "height") -> None:
    """Bird's-eye (top-down) view. x=forward, y=left. Colored by height/intensity."""
    x, y, z, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    c = z if color_by == "height" else inten
    cmap = "turbo" if color_by == "height" else "viridis"

    fig, ax = plt.subplots(figsize=(11, 11), facecolor="white")
    # forward (x) on vertical axis, left (y) on horizontal -> natural top-down
    sc = ax.scatter(-y, x, c=c, s=0.4, cmap=cmap, linewidths=0)
    ax.scatter([0], [0], c="red", s=120, marker="^",
               edgecolors="black", zorder=5, label="sensor (vehicle)")
    # foveation range rings
    for radius, lab in [(10, "5cm"), (30, "15cm"), (60, "30cm"), (100, "50cm")]:
        circ = plt.Circle((0, 0), radius, fill=False, ls="--",
                          color="gray", alpha=0.5, lw=0.8)
        ax.add_patch(circ)
        ax.text(0, radius, f" {radius}m", color="gray", fontsize=8, va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-15, 90)
    ax.set_xlabel("← left    |    right →   (meters)")
    ax.set_ylabel("forward (meters) →")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("height z (m)" if color_by == "height" else "intensity")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def side(pts: np.ndarray, title: str, out: Path) -> None:
    """Side elevation view (x forward vs z up) — shows the 2.5D height profile."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    fig, ax = plt.subplots(figsize=(13, 4), facecolor="white")
    sc = ax.scatter(x, z, c=z, s=0.4, cmap="turbo", linewidths=0)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("forward x (m)")
    ax.set_ylabel("height z (m)")
    ax.set_title(title + "  —  side elevation (the height a flat 2D map throws away)",
                 fontsize=12, weight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01).set_label("z (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def view3d(pts: np.ndarray, title: str, out: Path) -> None:
    """3D perspective view of the near field (r<40m for clarity)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    m = r < 40
    x, y, z = x[m], y[m], z[m]
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, s=0.5, cmap="turbo", linewidths=0)
    ax.set_xlabel("forward x (m)")
    ax.set_ylabel("left y (m)")
    ax.set_zlabel("up z (m)")
    ax.set_title(title + "  —  3D (near field r<40m)", fontsize=12, weight="bold")
    ax.view_init(elev=25, azim=-60)
    try:
        ax.set_box_aspect((2, 2, 0.4))
    except Exception:
        pass
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.05).set_label("z (m)")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def semantic_bev(pts: np.ndarray, title: str, out: Path) -> None:
    """BEV colored by the project's baseline semantic segmenter (real labels)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segmentation.baseline_segmenter import BaselineSegmenter
    from src.segmentation.classes import CLASS_ID

    seg = BaselineSegmenter(warmup=False)
    res = seg.segment(pts[:, :3])
    labels = res.labels
    x, y = pts[:, 0], pts[:, 1]

    colors = {
        CLASS_ID["ground"]: ("#2ca02c", "ground"),
        CLASS_ID["vehicle"]: ("#d62728", "vehicle"),
        CLASS_ID["pedestrian"]: ("#ffdd00", "pedestrian"),
        CLASS_ID["static_obstacle"]: ("#1f77b4", "static obstacle"),
        CLASS_ID["unknown"]: ("#7f7f7f", "unknown"),
    }
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="white")
    for cid, (col, lab) in colors.items():
        m = labels == cid
        if m.any():
            ax.scatter(-y[m], x[m], c=col, s=0.5, linewidths=0, label=f"{lab} ({m.sum()})")
    ax.scatter([0], [0], c="black", s=120, marker="^", zorder=5, label="sensor")
    ax.set_aspect("equal")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-15, 90)
    ax.set_xlabel("← left    |    right →   (m)")
    ax.set_ylabel("forward (m) →")
    ax.set_title(title + "  —  baseline semantic labels (PROTOTYPE)",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=9, markerscale=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def all_bev_grid(out: Path) -> None:
    """One small BEV thumbnail per frame, in a grid (overview of the whole drive)."""
    files = sorted(glob.glob(str(RAW / "*.bin")))
    n = len(files)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(files):
        pts = load(f)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        ax = axes[i]
        ax.scatter(-y, x, c=z, s=0.15, cmap="turbo", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-15, 80)
        ax.set_title(os.path.basename(f).replace(".bin", ""), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"KITTI drive 2011_09_26_drive_0001 — all {n} frames (BEV, height-colored)",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=int, default=0, help="frame index (0..N)")
    ap.add_argument("--all-bev", action="store_true",
                    help="render a grid thumbnail of every frame")
    ap.add_argument("--semantic", action="store_true",
                    help="also render a semantic-labeled BEV (slower: runs segmenter)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.all_bev:
        all_bev_grid(OUT / "all_frames_bev.png")
        return 0

    files = sorted(glob.glob(str(RAW / "*.bin")))
    if not files:
        print("No .bin frames in", RAW)
        return 1
    idx = max(0, min(args.frame, len(files) - 1))
    f = files[idx]
    pts = load(f)
    stem = os.path.basename(f).replace(".bin", "")
    title = f"KITTI frame {stem}  ({len(pts):,} pts)"

    bev(pts, title, OUT / f"{stem}_bev_height.png", color_by="height")
    bev(pts, title, OUT / f"{stem}_bev_intensity.png", color_by="intensity")
    side(pts, title, OUT / f"{stem}_side.png")
    view3d(pts, title, OUT / f"{stem}_3d.png")
    if args.semantic:
        semantic_bev(pts, title, OUT / f"{stem}_semantic.png")

    print("\nAll images in:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
