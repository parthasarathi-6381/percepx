"""The 2.5D cell representation.

A ``Cell`` is the serializable unit the dashboard consumes: one occupied cell of
a 2.5D map. It holds spatial position, aggregated elevation, occupancy, semantic
class, confidence, and point count.

Design note
-----------
Internally the grids do NOT store one ``Cell`` Python object per cell -- a full
5 cm grid can have millions of cells and per-object overhead would be far too
slow/heavy. Instead each grid keeps parallel NumPy arrays (structure-of-arrays)
and only materializes ``Cell`` objects (or a serializable dict) on demand for
occupied cells. ``Cell`` therefore defines the *schema* and the
(de)serialization format, not the internal storage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

# Sentinel used in class-id arrays for "no class assigned yet".
NO_CLASS = -1


@dataclass
class Cell:
    """A single occupied cell of a 2.5D map.

    Attributes
    ----------
    x, y:
        World-space position of the cell (meters). By convention this is the
        cell's lower corner origin: ``x = grid_x * resolution``. Which corner is
        documented per-grid; what matters is consistency.
    resolution:
        Cell size (meters) -- constant for a uniform grid, per-zone for a
        foveated grid. Stored so a mixed-resolution map is unambiguous.
    elevation:
        Aggregated height (meters). Max z by default (see config).
    occupancy:
        True if at least one valid point fell in the cell.
    semantic_class:
        Integer class id (see configs/config.yaml `classes`). ``NO_CLASS`` if
        unknown/unassigned.
    confidence:
        Aggregated classification confidence in [0, 1].
    point_count:
        Number of points aggregated into the cell.
    zone:
        Optional zone label ("near"/"mid"/"far"/"very_far"/"uniform").
    """

    x: float
    y: float
    resolution: float
    elevation: float
    occupancy: bool
    semantic_class: int
    confidence: float
    point_count: int
    zone: str = "uniform"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (for the dashboard / saving to disk)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cell":
        return cls(**d)
