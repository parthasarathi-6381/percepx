"""Baseline geometric semantic segmenter (a PROTOTYPE, not a neural network).

This lets the whole pipeline work end-to-end without training or downloading any
model. It is explicitly marked ``is_prototype = True`` so the dashboard never
claims deep-learning accuracy.

Method
------
1. **Ground** : fit a plane to the lowest points with a small RANSAC, then label
   points within a vertical threshold of that plane as ``ground``. RANSAC makes
   this robust to a mild road slope (vs a flat z-threshold).
2. **Cluster** : DBSCAN the remaining (non-ground) points into blobs.
3. **Classify each cluster** by simple, explainable geometry:
     - tall + small footprint          -> pedestrian
     - car-sized footprint + low height -> vehicle
     - very tall / large               -> static_obstacle
     - otherwise                        -> unknown
   Confidence reflects how cleanly the cluster matched its rule.

Everything here is deliberately simple and explainable (project rule 4). The
thresholds come from config so they are tunable without editing code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .base_segmenter import BaseSegmenter, SegmentationResult
from .classes import CLASS_ID

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Tunable thresholds for the baseline segmenter (from config.yaml)."""

    ground_z_threshold: float = 0.20        # |z - plane| below this -> ground
    ground_max_iterations: int = 100
    ground_distance_threshold: float = 0.20  # RANSAC inlier distance (m)
    cluster_eps: float = 0.5                 # DBSCAN neighborhood (m)
    cluster_min_points: int = 10
    # Cluster classification geometry (meters).
    ped_max_footprint: float = 1.0          # pedestrians are narrow
    ped_min_height: float = 0.8
    ped_max_height: float = 2.2             # people aren't taller than this
    vehicle_max_footprint: float = 6.0      # cars/vans footprint
    vehicle_max_height: float = 2.2

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BaselineConfig":
        b = cfg.get("segmentation", {}).get("baseline", {})
        return cls(
            ground_z_threshold=float(b.get("ground_z_threshold", 0.20)),
            ground_max_iterations=int(b.get("ground_max_iterations", 100)),
            ground_distance_threshold=float(b.get("ground_distance_threshold", 0.20)),
            cluster_eps=float(b.get("cluster_eps", 0.5)),
            cluster_min_points=int(b.get("cluster_min_points", 10)),
            ped_max_footprint=float(b.get("ped_max_footprint", 1.0)),
            ped_min_height=float(b.get("ped_min_height", 0.8)),
            ped_max_height=float(b.get("ped_max_height", 2.2)),
            vehicle_max_footprint=float(b.get("vehicle_max_footprint", 6.0)),
            vehicle_max_height=float(b.get("vehicle_max_height", 2.2)),
        )


class BaselineSegmenter(BaseSegmenter):
    """Geometric heuristic segmenter (prototype)."""

    name = "baseline_geometric"
    is_prototype = True

    def __init__(self, config: Optional[BaselineConfig] = None, seed: int = 0) -> None:
        self.config = config or BaselineConfig()
        self.seed = seed

    # ------------------------------------------------------------------
    def segment(self, points: np.ndarray) -> SegmentationResult:
        pts = np.asarray(points, dtype=np.float64)
        n = pts.shape[0]
        if n == 0:
            return SegmentationResult(
                labels=np.empty(0, dtype=np.int64),
                confidences=np.empty(0, dtype=np.float64),
                method=self.name,
                is_prototype=True,
                meta={"note": "empty cloud"},
            )

        labels = np.full(n, CLASS_ID["unknown"], dtype=np.int64)
        conf = np.full(n, 0.3, dtype=np.float64)  # low default confidence

        xyz = pts[:, :3]

        # --- 1) Ground via RANSAC plane fit ---
        ground_mask, plane = self._fit_ground(xyz)
        labels[ground_mask] = CLASS_ID["ground"]
        conf[ground_mask] = 0.9

        # --- 2) cluster non-ground points ---
        nonground_idx = np.where(~ground_mask)[0]
        cluster_meta: Dict[str, int] = {}
        if nonground_idx.size >= self.config.cluster_min_points:
            cl_labels = self._cluster(xyz[nonground_idx])
            # --- 3) classify each cluster ---
            for cid in np.unique(cl_labels):
                if cid == -1:  # DBSCAN noise -> leave as unknown
                    continue
                member = nonground_idx[cl_labels == cid]
                cls_id, c = self._classify_cluster(xyz[member])
                labels[member] = cls_id
                conf[member] = c

        meta = {
            "ground_points": int(ground_mask.sum()),
            "nonground_points": int((~ground_mask).sum()),
            "plane": None if plane is None else [float(v) for v in plane],
        }
        meta.update(cluster_meta)
        logger.info(
            "Baseline segmenter: %d ground / %d non-ground of %d points",
            meta["ground_points"], meta["nonground_points"], n,
        )
        return SegmentationResult(labels, conf, self.name, True, meta)

    # ------------------------------------------------------------------
    # Ground plane RANSAC (pure NumPy; no sklearn dependency for this part)
    # ------------------------------------------------------------------
    def _fit_ground(self, xyz: np.ndarray):
        """Return (ground_mask, plane) where plane = (a,b,c,d) for ax+by+cz+d=0.

        Seeds RANSAC from the low-z portion of the cloud so the road, not a roof,
        is fitted. Falls back to a flat z-threshold if RANSAC fails.
        """
        cfg = self.config
        n = xyz.shape[0]
        z = xyz[:, 2]

        # Candidate ground points: the lower part of the vertical extent.
        z_lo = np.percentile(z, 5)
        cand = np.where(z <= z_lo + 0.5)[0]  # within 0.5 m of the low band
        if cand.size < 3:
            # Not enough to fit a plane; use a simple z threshold.
            mask = z <= (z.min() + cfg.ground_z_threshold)
            return mask, None

        rng = np.random.default_rng(self.seed)
        best_inliers = None
        best_count = -1
        best_plane = None
        cand_xyz = xyz[cand]

        for _ in range(cfg.ground_max_iterations):
            sample = rng.choice(cand.size, size=3, replace=False)
            p0, p1, p2 = cand_xyz[sample]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -normal.dot(p0)
            # Distance of ALL points to this plane.
            dist = np.abs(xyz.dot(normal) + d)
            inliers = dist <= cfg.ground_distance_threshold
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d])

        if best_inliers is None:
            mask = z <= (z.min() + cfg.ground_z_threshold)
            return mask, None

        # Only accept as ground if the plane is roughly horizontal (normal ~ +z),
        # otherwise we may have fit a wall. Fall back to z-threshold if not.
        if best_plane is not None and abs(best_plane[2]) < 0.7:
            mask = z <= (z.min() + cfg.ground_z_threshold)
            return mask, None

        return best_inliers, best_plane

    # ------------------------------------------------------------------
    def _cluster(self, xyz: np.ndarray) -> np.ndarray:
        """DBSCAN cluster labels for non-ground points (-1 = noise)."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:  # pragma: no cover
            logger.warning("scikit-learn not available; skipping clustering.")
            return np.full(xyz.shape[0], -1, dtype=np.int64)

        db = DBSCAN(
            eps=self.config.cluster_eps,
            min_samples=self.config.cluster_min_points,
        )
        return db.fit_predict(xyz)

    # ------------------------------------------------------------------
    def _classify_cluster(self, member_xyz: np.ndarray):
        """Classify one cluster by simple geometry -> (class_id, confidence)."""
        cfg = self.config
        mins = member_xyz.min(axis=0)
        maxs = member_xyz.max(axis=0)
        extent = maxs - mins
        footprint = float(max(extent[0], extent[1]))  # largest horizontal span
        height = float(extent[2])

        # tall/large structure -> static obstacle (checked FIRST so a tall thin
        # pole is not mistaken for a pedestrian).
        if height > cfg.vehicle_max_height or footprint > cfg.vehicle_max_footprint:
            return CLASS_ID["static_obstacle"], 0.55
        # pedestrian: narrow footprint, human-height band
        if (
            footprint <= cfg.ped_max_footprint
            and cfg.ped_min_height <= height <= cfg.ped_max_height
        ):
            return CLASS_ID["pedestrian"], 0.6
        # vehicle: car-sized footprint, not too tall
        if footprint <= cfg.vehicle_max_footprint and height <= cfg.vehicle_max_height:
            return CLASS_ID["vehicle"], 0.6
        return CLASS_ID["unknown"], 0.3
