"""Semantic class registry.

Single source of truth for the mapping between class names, integer ids, and
display colors. Ids MUST match ``configs/config.yaml`` and are used everywhere
(segmenters, grids, visualization).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Canonical order -> integer id. Kept in sync with config.yaml `classes`.
CLASS_NAMES: List[str] = [
    "ground",           # 0
    "vehicle",          # 1
    "pedestrian",       # 2
    "static_obstacle",  # 3
    "unknown",          # 4
]

CLASS_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Default colors (overridable from config). Consistent across all plots.
DEFAULT_COLORS: Dict[str, str] = {
    "ground": "#2ca02c",           # green
    "vehicle": "#d62728",          # red
    "pedestrian": "#ffdd00",       # yellow
    "static_obstacle": "#1f77b4",  # blue
    "unknown": "#7f7f7f",          # gray
}

NUM_CLASSES: int = len(CLASS_NAMES)


@dataclass
class ClassRegistry:
    """Resolved class metadata (names/ids/colors), possibly from config."""

    names: List[str]
    colors: Dict[str, str]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ClassRegistry":
        classes = cfg.get("classes", {})
        if not classes:
            return cls(list(CLASS_NAMES), dict(DEFAULT_COLORS))
        # Order by declared id so integer ids are stable.
        items = sorted(classes.items(), key=lambda kv: kv[1].get("id", 0))
        names = [name for name, _ in items]
        colors = {name: meta.get("color", DEFAULT_COLORS.get(name, "#7f7f7f"))
                  for name, meta in items}
        return cls(names, colors)

    def id_of(self, name: str) -> int:
        return self.names.index(name)

    def name_of(self, cid: int) -> str:
        if 0 <= cid < len(self.names):
            return self.names[cid]
        return "unknown"

    def color_list(self) -> List[str]:
        return [self.colors[n] for n in self.names]

    @property
    def num_classes(self) -> int:
        return len(self.names)
