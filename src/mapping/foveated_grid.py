"""Foveated 2.5D grid -- the core innovation.

Points are split by radial distance into four resolution zones and each zone is
mapped by its own ``UniformGrid`` at the zone's cell size:

    NearGrid     0-10 m   @ 5  cm
    MidGrid      10-30 m  @ 15 cm
    FarGrid      30-60 m  @ 30 cm
    VeryFarGrid  60-100 m @ 50 cm

Why separate per-zone grids?
----------------------------
Each zone has a different cell size, so their integer cell indices live in
*different lattices*. Keeping one ``UniformGrid`` per zone means:

* We reuse the already-tested aggregation code path exactly.
* Cell coordinates within a zone are unambiguous (all share that zone's res).
* There is no rounding/aliasing between resolutions inside a single array.

Boundary handling (how data loss is avoided)
--------------------------------------------
``ResolutionPolicy.assign_zones`` assigns every point to EXACTLY ONE zone using
inclusive upper edges. A point is never sent to two zones and never dropped at a
boundary. The only intentional drop is ``r > max_range`` (zone index -1), which
is counted and reported -- never silent. Because zones tile ``[0, max_range]``
with no gaps and no overlap, the union of the four grids covers every retained
point once.

The four zones' cells simply coexist in the output map, each tagged with its
zone name and resolution, so the dashboard can render the changing cell size.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .cell import Cell
from .resolution_policy import ResolutionPolicy
from .uniform_grid import UniformGrid

logger = logging.getLogger(__name__)


class FoveatedGrid:
    """Manages the four per-zone grids and routes points by distance."""

    def __init__(
        self,
        policy: ResolutionPolicy,
        height_aggregation: str = "max",
        confidence_aggregation: str = "mean",
        n_classes: int = 5,
    ) -> None:
        self.policy = policy
        self.n_classes = n_classes
        # One UniformGrid per zone, at that zone's resolution.
        self.zone_grids: Dict[str, UniformGrid] = {
            z.name: UniformGrid(
                resolution=z.resolution,
                height_aggregation=height_aggregation,
                confidence_aggregation=confidence_aggregation,
                zone_label=z.name,
            )
            for z in policy.zones
        }
        # Book-keeping so nothing is silently discarded.
        self.discarded_out_of_range: int = 0
        self._points_per_zone: Dict[str, int] = {z.name: 0 for z in policy.zones}

    # ------------------------------------------------------------------
    def insert(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
    ) -> None:
        """Route points to zones by radial distance, then insert per zone."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] == 0:
            return
        n = pts.shape[0]

        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        zone_idx = self.policy.assign_zones(r)  # 0..3 or -1 (discard)

        self.discarded_out_of_range += int((zone_idx == -1).sum())

        for zi, zone in enumerate(self.policy.zones):
            mask = zone_idx == zi
            count = int(mask.sum())
            self._points_per_zone[zone.name] += count
            if count == 0:
                continue
            zl = labels[mask] if labels is not None else None
            zc = confidences[mask] if confidences is not None else None
            self.zone_grids[zone.name].insert(
                pts[mask], labels=zl, confidences=zc, n_classes=self.n_classes
            )

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------
    def num_cells(self) -> int:
        """Total occupied cells across all zones."""
        return sum(g.num_cells() for g in self.zone_grids.values())

    def num_cells_by_zone(self) -> Dict[str, int]:
        return {name: g.num_cells() for name, g in self.zone_grids.items()}

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Concatenated occupied-cell arrays across all zones (for viz).

        Adds a ``zone_index`` array (0..3) so visualizations can color by zone.
        """
        parts: List[Dict[str, np.ndarray]] = []
        zone_indices: List[np.ndarray] = []
        for zi, zone in enumerate(self.policy.zones):
            g = self.zone_grids[zone.name]
            if g.num_cells() == 0:
                continue
            a = g.to_arrays()
            parts.append(a)
            zone_indices.append(np.full(g.num_cells(), zi, dtype=np.int64))

        if not parts:
            empty = np.empty(0)
            return {
                "x": empty, "y": empty, "elevation": empty,
                "semantic_class": np.empty(0, dtype=np.int64),
                "confidence": empty, "point_count": np.empty(0, dtype=np.int64),
                "resolution": empty, "zone_index": np.empty(0, dtype=np.int64),
            }

        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].keys()}
        out["zone_index"] = np.concatenate(zone_indices)
        return out

    def cells(self) -> List[Cell]:
        """All occupied cells across zones as serializable ``Cell`` objects."""
        out: List[Cell] = []
        for zone in self.policy.zones:
            out.extend(self.zone_grids[zone.name].cells())
        return out

    def stats(self) -> Dict[str, Any]:
        """Summary stats used by the benchmark and dashboard."""
        by_zone = {}
        for zone in self.policy.zones:
            g = self.zone_grids[zone.name]
            by_zone[zone.name] = {
                "resolution": zone.resolution,
                "num_cells": g.num_cells(),
                "points": self._points_per_zone[zone.name],
            }
        return {
            "num_cells": self.num_cells(),
            "discarded_out_of_range": self.discarded_out_of_range,
            "by_zone": by_zone,
        }
