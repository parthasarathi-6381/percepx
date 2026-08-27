"""Distance-dependent resolution policy -- the heart of the foveated idea.

A ``ResolutionPolicy`` maps a radial distance ``r = sqrt(x^2 + y^2)`` to a
resolution zone (near / mid / far / very_far) and its cell size. All numbers
come from ``config.yaml``; nothing is hard-coded in the mapping engine.

Boundary convention (documented + tested)
-----------------------------------------
Upper edges are INCLUSIVE, matching the spec:

    r <= near_range           -> near      (default 0.05 m)
    near_range < r <= mid_range  -> mid       (default 0.15 m)
    mid_range  < r <= far_range  -> far       (default 0.30 m)
    far_range  < r <= max_range  -> very_far  (default 0.50 m)
    r > max_range             -> DISCARD

So a point exactly at 10 m is "near", exactly at 30 m is "mid", exactly at
60 m is "far", exactly at 100 m is "very_far", and anything beyond 100 m is
discarded. This makes zone membership unambiguous at the edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# Canonical zone names, in near->far order.
ZONE_NAMES: Tuple[str, ...] = ("near", "mid", "far", "very_far")


@dataclass(frozen=True)
class Zone:
    """One resolution zone: name, [r_lo, r_hi] radial band, and cell size."""

    name: str
    r_lo: float          # exclusive lower bound (except near, whose lower is 0)
    r_hi: float          # inclusive upper bound
    resolution: float    # meters per cell


class ResolutionPolicy:
    """Config-driven mapping from radial distance -> zone / resolution."""

    def __init__(
        self,
        near_range: float = 10.0,
        mid_range: float = 30.0,
        far_range: float = 60.0,
        max_range: float = 100.0,
        res_near: float = 0.05,
        res_mid: float = 0.15,
        res_far: float = 0.30,
        res_very_far: float = 0.50,
    ) -> None:
        # Validate monotonic, positive ranges.
        ranges = [near_range, mid_range, far_range, max_range]
        if any(a >= b for a, b in zip(ranges, ranges[1:])):
            raise ValueError(
                "ranges must be strictly increasing: "
                f"near<mid<far<max, got {ranges}"
            )
        self.max_range = float(max_range)
        self.zones: List[Zone] = [
            Zone("near", 0.0, float(near_range), float(res_near)),
            Zone("mid", float(near_range), float(mid_range), float(res_mid)),
            Zone("far", float(mid_range), float(far_range), float(res_far)),
            Zone("very_far", float(far_range), float(max_range), float(res_very_far)),
        ]
        # Upper edges used for vectorized bucketing.
        self._edges = np.array([z.r_hi for z in self.zones], dtype=np.float64)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ResolutionPolicy":
        """Build from the full config dict (uses `mapping` and `resolution`)."""
        m = cfg.get("mapping", {})
        r = cfg.get("resolution", {})
        return cls(
            near_range=float(m.get("near_range", 10.0)),
            mid_range=float(m.get("mid_range", 30.0)),
            far_range=float(m.get("far_range", 60.0)),
            max_range=float(m.get("max_range", 100.0)),
            res_near=float(r.get("near", 0.05)),
            res_mid=float(r.get("mid", 0.15)),
            res_far=float(r.get("far", 0.30)),
            res_very_far=float(r.get("very_far", 0.50)),
        )

    # ------------------------------------------------------------------
    # Scalar helpers (readable; used in docs/tests)
    # ------------------------------------------------------------------
    def zone_for_distance(self, r: float) -> str | None:
        """Return the zone name for a single distance, or ``None`` if discarded."""
        if r < 0:
            raise ValueError("radial distance cannot be negative")
        if r > self.max_range:
            return None
        for z in self.zones:
            if r <= z.r_hi:
                return z.name
        return None  # unreachable given r <= max_range

    def resolution_for_distance(self, r: float) -> float | None:
        """Return the cell size for a single distance, or ``None`` if discarded."""
        name = self.zone_for_distance(r)
        if name is None:
            return None
        return self.resolution_of(name)

    def resolution_of(self, zone_name: str) -> float:
        for z in self.zones:
            if z.name == zone_name:
                return z.resolution
        raise KeyError(f"unknown zone: {zone_name}")

    # ------------------------------------------------------------------
    # Vectorized bucketing (the fast path)
    # ------------------------------------------------------------------
    def assign_zones(self, r: np.ndarray) -> np.ndarray:
        """Vectorized: map an array of distances to zone indices.

        Returns an int array where each entry is 0..3 for near..very_far, or
        ``-1`` for "discard" (r > max_range). Negative distances are invalid.
        """
        r = np.asarray(r, dtype=np.float64)
        if (r < 0).any():
            raise ValueError("radial distance cannot be negative")
        # searchsorted with side="left" on inclusive upper edges:
        #   r <= edges[i] picks the first bucket whose upper edge is >= r.
        # Using side="left" makes r exactly equal to an edge fall INTO that
        # zone (inclusive upper bound), which matches the spec.
        idx = np.searchsorted(self._edges, r, side="left")
        # Anything with idx == len(edges) is beyond max_range -> discard (-1).
        idx = np.where(idx >= len(self._edges), -1, idx)
        return idx.astype(np.int64)

    def zone_name(self, index: int) -> str:
        return ZONE_NAMES[index]

    def summary(self) -> List[Dict[str, Any]]:
        """Human-readable zone table (for logging / dashboard)."""
        rows = []
        for z in self.zones:
            rows.append(
                {
                    "zone": z.name,
                    "range_m": f"{z.r_lo:g}-{z.r_hi:g}",
                    "resolution_m": z.resolution,
                    "resolution_cm": round(z.resolution * 100, 1),
                }
            )
        return rows
