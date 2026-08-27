"""LiDAR loading and preprocessing.

This module is deliberately self-contained and modular:

    LidarProcessor.load(path)        -> PointCloud          (Phase 2)
    LidarProcessor.preprocess(pc)    -> (PointCloud, stats) (Phase 3)

A ``PointCloud`` is a thin wrapper around a single ``(N, 4)`` float32 NumPy
array holding ``(x, y, z, intensity)`` for each point, plus convenience
accessors. Keeping the point cloud as one contiguous array keeps the later
vectorized mapping code fast and simple.

Design rules honored here:
* Never silently discard points -- every filter records how many points it
  removed in a ``PreprocessStats`` object.
* All parameters are configurable (via ``PreprocessConfig``); nothing is
  hard-coded in the algorithms.
* Works with negative x/y (the sensor is at the origin; points surround it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Column indices into the (N, 4) point array.
X, Y, Z, INTENSITY = 0, 1, 2, 3


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass
class PointCloud:
    """A LiDAR point cloud: ``points`` is an ``(N, 4)`` float32 array.

    Columns are (x, y, z, intensity). Kept as a single array so that all
    downstream mapping code can vectorize over it cleanly.
    """

    points: np.ndarray

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 4:
            raise ValueError(
                f"PointCloud expects an (N, 4) array, got shape {pts.shape}"
            )
        self.points = pts

    # --- convenient column views (no copy) ---
    @property
    def x(self) -> np.ndarray:
        return self.points[:, X]

    @property
    def y(self) -> np.ndarray:
        return self.points[:, Y]

    @property
    def z(self) -> np.ndarray:
        return self.points[:, Z]

    @property
    def intensity(self) -> np.ndarray:
        return self.points[:, INTENSITY]

    @property
    def size(self) -> int:
        """Number of points."""
        return int(self.points.shape[0])

    def radial_distance(self) -> np.ndarray:
        """Return per-point horizontal distance ``r = sqrt(x^2 + y^2)``.

        This is the 2D distance used for range filtering and for choosing the
        foveated resolution zone. Height (z) is intentionally excluded.
        """
        return np.sqrt(self.x**2 + self.y**2)

    def __len__(self) -> int:  # so len(pc) works
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PointCloud(N={self.size})"


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing. Mirrors the ``preprocessing`` block
    of ``config.yaml`` but decoupled from YAML so it can be built in code/tests.
    """

    min_range: float = 0.0
    max_range: float = 100.0
    z_min: float = -3.0
    z_max: float = 3.0
    remove_invalid: bool = True
    voxel_downsample: bool = False
    voxel_size: float = 0.10

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "PreprocessConfig":
        """Build from the ``preprocessing`` sub-dict of the loaded config."""
        vd = cfg.get("voxel_downsample", {}) or {}
        return cls(
            min_range=float(cfg.get("min_range", 0.0)),
            max_range=float(cfg.get("max_range", 100.0)),
            z_min=float(cfg.get("z_min", -3.0)),
            z_max=float(cfg.get("z_max", 3.0)),
            remove_invalid=bool(cfg.get("remove_invalid", True)),
            voxel_downsample=bool(vd.get("enabled", False)),
            voxel_size=float(vd.get("voxel_size", 0.10)),
        )


@dataclass
class PreprocessStats:
    """Book-keeping so no point is silently discarded.

    Every stage records how many points it removed. ``as_dict`` gives a
    JSON/log-friendly summary.
    """

    input_points: int = 0
    removed_invalid: int = 0
    removed_out_of_range: int = 0
    removed_z_crop: int = 0
    removed_voxel: int = 0
    output_points: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return (
            self.removed_invalid
            + self.removed_out_of_range
            + self.removed_z_crop
            + self.removed_voxel
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_points": self.input_points,
            "removed_invalid": self.removed_invalid,
            "removed_out_of_range": self.removed_out_of_range,
            "removed_z_crop": self.removed_z_crop,
            "removed_voxel": self.removed_voxel,
            "total_removed": self.total_removed,
            "output_points": self.output_points,
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------------
# Loader + preprocessor
# ----------------------------------------------------------------------------
class LidarProcessor:
    """Loads KITTI ``.bin`` frames and applies modular preprocessing."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    # ---- Phase 2: loading -------------------------------------------------
    @staticmethod
    def load(path: str | Path) -> PointCloud:
        """Load a single KITTI Velodyne ``.bin`` frame.

        The KITTI format is a flat little-endian ``float32`` buffer laid out as
        ``[x0, y0, z0, i0, x1, y1, z1, i1, ...]`` -- i.e. 4 floats per point.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file's byte count is not a multiple of 4 floats.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"LiDAR file not found: {p}")

        raw = np.fromfile(str(p), dtype=np.float32)
        if raw.size == 0:
            logger.warning("Loaded an EMPTY LiDAR file: %s", p)
            return PointCloud(np.empty((0, 4), dtype=np.float32))

        if raw.size % 4 != 0:
            raise ValueError(
                f"{p} has {raw.size} floats, not divisible by 4 "
                "(expected KITTI x,y,z,intensity layout)."
            )

        points = raw.reshape(-1, 4)
        logger.info("Loaded %d points from %s", points.shape[0], p.name)
        return PointCloud(points)

    @staticmethod
    def save(pc: PointCloud, path: str | Path) -> None:
        """Write a point cloud back out in KITTI ``.bin`` layout (handy for
        creating small deterministic test frames)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pc.points.astype(np.float32).tofile(str(p))
        logger.info("Wrote %d points -> %s", pc.size, p)

    # ---- Phase 3: preprocessing ------------------------------------------
    def preprocess(self, pc: PointCloud) -> tuple[PointCloud, PreprocessStats]:
        """Run the full modular preprocessing chain.

        Order: remove invalid -> range filter -> z crop -> optional voxel
        downsample. Each stage updates ``stats``.

        Returns the cleaned ``PointCloud`` and a ``PreprocessStats`` record.
        """
        stats = PreprocessStats(input_points=pc.size)

        if pc.size == 0:
            stats.notes.append("empty input cloud")
            stats.output_points = 0
            return pc, stats

        pts = pc.points

        # 1) Remove invalid (NaN / Inf) points.
        if self.config.remove_invalid:
            pts, removed = self._remove_invalid(pts)
            stats.removed_invalid = removed

        # 2) Radial range filter (uses horizontal distance).
        pts, removed = self._filter_range(pts)
        stats.removed_out_of_range = removed

        # 3) Vertical (z) crop.
        pts, removed = self._filter_z(pts)
        stats.removed_z_crop = removed

        # 4) Optional voxel downsampling.
        if self.config.voxel_downsample:
            before = pts.shape[0]
            pts = self._voxel_downsample(pts, self.config.voxel_size)
            stats.removed_voxel = before - pts.shape[0]

        out = PointCloud(pts)
        stats.output_points = out.size
        logger.info(
            "Preprocess: %d -> %d points (removed %d: invalid=%d, range=%d, "
            "z=%d, voxel=%d)",
            stats.input_points,
            stats.output_points,
            stats.total_removed,
            stats.removed_invalid,
            stats.removed_out_of_range,
            stats.removed_z_crop,
            stats.removed_voxel,
        )
        return out, stats

    # ---- individual filters (pure functions on arrays) -------------------
    @staticmethod
    def _remove_invalid(pts: np.ndarray) -> tuple[np.ndarray, int]:
        """Drop rows containing any NaN or Inf in x/y/z/intensity."""
        finite = np.isfinite(pts).all(axis=1)
        removed = int((~finite).sum())
        return pts[finite], removed

    def _filter_range(self, pts: np.ndarray) -> tuple[np.ndarray, int]:
        """Keep points with ``min_range <= r <= max_range`` (r = sqrt(x^2+y^2)).

        Works for negative x/y because ``r`` is always non-negative.
        """
        r = np.sqrt(pts[:, X] ** 2 + pts[:, Y] ** 2)
        keep = (r >= self.config.min_range) & (r <= self.config.max_range)
        removed = int((~keep).sum())
        return pts[keep], removed

    def _filter_z(self, pts: np.ndarray) -> tuple[np.ndarray, int]:
        """Keep points with ``z_min <= z <= z_max``."""
        keep = (pts[:, Z] >= self.config.z_min) & (pts[:, Z] <= self.config.z_max)
        removed = int((~keep).sum())
        return pts[keep], removed

    @staticmethod
    def _voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
        """Simple voxel-grid downsampling: keep the centroid of each occupied
        voxel. Pure NumPy so we don't hard-depend on Open3D.

        Deterministic: voxel keys are computed with floor division, and the
        centroid of the points in each voxel is returned (order sorted by
        voxel key for reproducibility).
        """
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        if pts.shape[0] == 0:
            return pts

        # Integer voxel index per point (works with negative coords via floor).
        keys = np.floor(pts[:, :3] / voxel_size).astype(np.int64)
        # Map each unique voxel to a group and average the points inside it.
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        inverse = inverse.ravel()
        n_voxels = counts.shape[0]
        sums = np.zeros((n_voxels, pts.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, pts)
        centroids = (sums / counts[:, None]).astype(np.float32)
        return centroids
