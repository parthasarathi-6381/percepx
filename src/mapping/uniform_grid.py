"""Uniform-resolution 2.5D grid (the conventional baseline).

Every point is projected to a single fixed-resolution 2D grid. Each occupied
cell aggregates elevation, occupancy, semantic class, confidence, and point
count. This is the baseline the foveated grid is benchmarked against.

Coordinate convention
---------------------
    grid_x = floor(x / resolution)
    grid_y = floor(y / resolution)

``floor`` (not ``int()``) is used so negative coordinates map correctly:
``floor(-0.01 / 0.05) = floor(-0.2) = -1`` -- a point just left of the origin
lands in cell -1, not 0. The sensor sits at the origin and points surround it,
so negative indices are normal and expected.

Storage
-------
Cells are stored **sparsely**: only occupied cells consume memory. A dense array
over +/-100 m at 5 cm would be 4000x4000 = 16M cells even though a single scan
occupies a small fraction of them. Aggregation is fully vectorized with a
``np.unique`` group-by, so latency reflects real mapping cost (important for the
benchmark) rather than Python-loop overhead.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cell import NO_CLASS, Cell

logger = logging.getLogger(__name__)


class UniformGrid:
    """A single-resolution 2.5D occupancy/elevation/semantic grid."""

    def __init__(
        self,
        resolution: float = 0.05,
        height_aggregation: str = "max",
        confidence_aggregation: str = "mean",
        zone_label: str = "uniform",
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        if height_aggregation not in ("max", "mean", "min"):
            raise ValueError(f"unknown height_aggregation: {height_aggregation}")
        if confidence_aggregation not in ("mean", "max"):
            raise ValueError(f"unknown confidence_aggregation: {confidence_aggregation}")

        self.resolution = float(resolution)
        self.height_aggregation = height_aggregation
        self.confidence_aggregation = confidence_aggregation
        self.zone_label = zone_label

        # Resolved per-cell arrays (parallel; index i is one occupied cell).
        # Populated by insert(). Empty until then.
        self._gx = np.empty(0, dtype=np.int64)   # cell index x
        self._gy = np.empty(0, dtype=np.int64)   # cell index y
        self._elevation = np.empty(0, dtype=np.float64)
        self._semantic = np.empty(0, dtype=np.int64)
        self._confidence = np.empty(0, dtype=np.float64)
        self._count = np.empty(0, dtype=np.int64)
        self._n_classes = 5  # number of semantic classes (for vote matrix)

    # ------------------------------------------------------------------
    # Coordinate math
    # ------------------------------------------------------------------
    def cell_index(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized ``floor(x/res), floor(y/res)`` -> int64 arrays.

        Works for negative coordinates (uses floor, not truncation).
        """
        gx = np.floor(np.asarray(x, dtype=np.float64) / self.resolution).astype(np.int64)
        gy = np.floor(np.asarray(y, dtype=np.float64) / self.resolution).astype(np.int64)
        return gx, gy

    # ------------------------------------------------------------------
    # Insertion / aggregation (vectorized group-by)
    # ------------------------------------------------------------------
    def insert(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        n_classes: int = 5,
    ) -> None:
        """Insert points, aggregating per cell. Replaces any previous contents.

        Parameters
        ----------
        points:
            ``(N, >=3)`` array; columns 0,1,2 are x, y, z. Extra columns ignored.
        labels:
            Optional ``(N,)`` integer class ids (0..n_classes-1). ``None`` ->
            all cells get ``NO_CLASS``.
        confidences:
            Optional ``(N,)`` floats in [0,1]. Defaults to 1.0.
        n_classes:
            Number of semantic classes (defines the vote matrix width).

        Aggregation
        -----------
        * elevation  : max z (default) / mean / min
        * occupancy  : any point -> True
        * class      : argmax of per-class summed confidence (vote-by-confidence)
        * confidence : mean (default) or max of the winning cell's points
        * point_count: number of points in the cell
        """
        self._n_classes = int(n_classes)
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] == 0:
            self._reset_empty()
            return
        n = pts.shape[0]

        if labels is None:
            labels = np.full(n, NO_CLASS, dtype=np.int64)
        else:
            labels = np.asarray(labels).astype(np.int64)
        if confidences is None:
            confidences = np.ones(n, dtype=np.float64)
        else:
            confidences = np.asarray(confidences, dtype=np.float64)

        gx, gy = self.cell_index(pts[:, 0], pts[:, 1])
        z = pts[:, 2]

        # Group points by unique (gx, gy) cell.
        #
        # Pack (gx, gy) into a single int64 key so we can use the FAST 1-D
        # np.unique path (sorting one integer column) instead of the much slower
        # np.unique(axis=0) which lexsorts a 2-column array. Indices can be
        # negative (sensor at origin), so we offset by the per-axis minimum into
        # a non-negative range, then bit-pack. This is exact and reversible.
        gx_min = int(gx.min())
        gy_min = int(gy.min())
        ux = (gx - gx_min).astype(np.int64)   # >= 0
        uy = (gy - gy_min).astype(np.int64)   # >= 0
        span_y = int(uy.max()) + 1            # number of distinct y rows + 1
        packed = ux * span_y + uy             # unique per (gx, gy)

        uniq_keys, inverse = np.unique(packed, return_inverse=True)
        inverse = inverse.ravel()
        m = uniq_keys.shape[0]  # number of occupied cells

        # Unpack the winning keys back to integer cell indices.
        self._gx = (uniq_keys // span_y).astype(np.int64) + gx_min
        self._gy = (uniq_keys % span_y).astype(np.int64) + gy_min

        # point_count per cell
        self._count = np.bincount(inverse, minlength=m).astype(np.int64)

        # --- elevation ---
        if self.height_aggregation == "mean":
            zsum = np.bincount(inverse, weights=z, minlength=m)
            self._elevation = zsum / self._count
        elif self.height_aggregation == "min":
            self._elevation = self._reduce(inverse, z, m, np.minimum, +np.inf)
        else:  # max
            self._elevation = self._reduce(inverse, z, m, np.maximum, -np.inf)

        # --- semantic class: per-cell per-class summed confidence, then argmax ---
        valid = labels != NO_CLASS
        if valid.any():
            # Build a (m, n_classes) matrix of summed confidence via flat index.
            cls = labels[valid]
            cell_idx = inverse[valid]
            conf = confidences[valid]
            flat = cell_idx * self._n_classes + cls
            vote = np.bincount(
                flat, weights=conf, minlength=m * self._n_classes
            ).reshape(m, self._n_classes)
            has_vote = vote.sum(axis=1) > 0
            winner = np.argmax(vote, axis=1)
            self._semantic = np.where(has_vote, winner, NO_CLASS).astype(np.int64)
        else:
            self._semantic = np.full(m, NO_CLASS, dtype=np.int64)

        # --- confidence readout ---
        if self.confidence_aggregation == "max":
            self._confidence = self._reduce(inverse, confidences, m, np.maximum, -np.inf)
        else:  # mean
            csum = np.bincount(inverse, weights=confidences, minlength=m)
            self._confidence = csum / self._count

    @staticmethod
    def _reduce(inverse, values, m, ufunc, init) -> np.ndarray:
        """Scatter-reduce (max/min) of ``values`` grouped by ``inverse``."""
        out = np.full(m, init, dtype=np.float64)
        ufunc.at(out, inverse, values)
        return out

    def _reset_empty(self) -> None:
        self._gx = np.empty(0, dtype=np.int64)
        self._gy = np.empty(0, dtype=np.int64)
        self._elevation = np.empty(0, dtype=np.float64)
        self._semantic = np.empty(0, dtype=np.int64)
        self._confidence = np.empty(0, dtype=np.float64)
        self._count = np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------
    def num_cells(self) -> int:
        """Number of occupied cells."""
        return int(self._gx.shape[0])

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Occupied-cell data as parallel NumPy arrays (for fast viz/benchmark).

        Keys: x, y, elevation, semantic_class, confidence, point_count, zone.
        x/y are world coordinates of the cell origin (gx*res, gy*res).
        """
        return {
            "x": self._gx * self.resolution,
            "y": self._gy * self.resolution,
            "elevation": self._elevation.copy(),
            "semantic_class": self._semantic.copy(),
            "confidence": self._confidence.copy(),
            "point_count": self._count.copy(),
            "resolution": np.full(self.num_cells(), self.resolution),
        }

    def cells(self) -> List[Cell]:
        """Materialize all occupied cells as ``Cell`` objects (serializable)."""
        out: List[Cell] = []
        for i in range(self.num_cells()):
            out.append(
                Cell(
                    x=float(self._gx[i] * self.resolution),
                    y=float(self._gy[i] * self.resolution),
                    resolution=self.resolution,
                    elevation=float(self._elevation[i]),
                    occupancy=True,
                    semantic_class=int(self._semantic[i]),
                    confidence=float(self._confidence[i]),
                    point_count=int(self._count[i]),
                    zone=self.zone_label,
                )
            )
        return out

    def stats(self) -> Dict[str, float]:
        """Summary stats used by the benchmark."""
        return {
            "resolution": self.resolution,
            "num_cells": self.num_cells(),
            "total_points": int(self._count.sum()) if self.num_cells() else 0,
        }
