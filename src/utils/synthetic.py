"""Deterministic synthetic LiDAR frame generator.

Lets the pipeline run and be tested end-to-end WITHOUT downloading KITTI. The
generated scene is intentionally simple and physically plausible so the
baseline segmenter and both grids have meaningful structure to work on:

* a flat ground plane (large, low z, spanning the full range)
* a few box-like "vehicles" (clusters at car height)
* a couple of thin tall "pedestrians"
* some scattered "static obstacle" points (e.g. poles / walls)

Everything is seeded, so the frame is byte-for-byte reproducible. This is NOT
real data and is clearly labeled as synthetic wherever it is produced.
"""
from __future__ import annotations

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_frame(seed: int = 42, n_ground: int = 60_000) -> np.ndarray:
    """Return an ``(N, 4)`` float32 KITTI-style array (x, y, z, intensity).

    Deterministic given ``seed``.
    """
    rng = _rng(seed)
    parts = []

    # --- Ground plane: spread across +/- range, small z noise around -1.7 m ---
    # KITTI Velodyne sits ~1.7 m above the road, so ground z ~ -1.7 m.
    gr = rng.uniform(1.0, 95.0, size=n_ground)          # radial distance
    gth = rng.uniform(0, 2 * np.pi, size=n_ground)       # angle
    gx = gr * np.cos(gth)
    gy = gr * np.sin(gth)
    gz = -1.7 + rng.normal(0, 0.02, size=n_ground)       # nearly flat
    gi = rng.uniform(0.0, 0.3, size=n_ground)
    parts.append(np.column_stack([gx, gy, gz, gi]))

    # --- Vehicles: box clusters at a few locations, z in [-1.5, 0.0] ---
    vehicle_centers = [(8, 3), (15, -6), (35, 10), (55, -20)]
    for (cx, cy) in vehicle_centers:
        n = 1500
        vx = cx + rng.uniform(-2.0, 2.0, size=n)
        vy = cy + rng.uniform(-0.9, 0.9, size=n)
        vz = rng.uniform(-1.5, 0.2, size=n)              # car body height
        vi = rng.uniform(0.3, 0.6, size=n)
        parts.append(np.column_stack([vx, vy, vz, vi]))

    # --- Pedestrians: thin tall clusters, z up to ~0.2 m ---
    ped_centers = [(6, -2), (12, 4)]
    for (cx, cy) in ped_centers:
        n = 400
        px = cx + rng.uniform(-0.3, 0.3, size=n)
        py = cy + rng.uniform(-0.3, 0.3, size=n)
        pz = rng.uniform(-1.6, 0.2, size=n)              # standing person
        pi = rng.uniform(0.2, 0.5, size=n)
        parts.append(np.column_stack([px, py, pz, pi]))

    # --- Static obstacles: poles / wall segments, taller z ---
    obs_centers = [(20, 20), (40, -15), (70, 5)]
    for (cx, cy) in obs_centers:
        n = 800
        ox = cx + rng.uniform(-0.5, 0.5, size=n)
        oy = cy + rng.uniform(-0.5, 0.5, size=n)
        oz = rng.uniform(-1.5, 2.0, size=n)              # tall structure
        oi = rng.uniform(0.4, 0.8, size=n)
        parts.append(np.column_stack([ox, oy, oz, oi]))

    cloud = np.concatenate(parts, axis=0).astype(np.float32)
    # Shuffle so classes are interleaved (like a real scan ordering-agnostic set)
    rng.shuffle(cloud)
    return cloud


def write_frame(path: str, seed: int = 42) -> str:
    """Generate a synthetic frame and write it as a KITTI ``.bin`` file."""
    from pathlib import Path

    cloud = generate_frame(seed=seed)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cloud.tofile(str(p))
    return str(p)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Generate a synthetic KITTI-style .bin frame")
    ap.add_argument("--out", default="data/raw/synthetic_000000.bin")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = write_frame(args.out, seed=args.seed)
    print(f"[synthetic] wrote frame -> {out}")
