"""Build semantic 2.5D maps by combining points with per-point labels.

This is the small seam that joins segmentation to mapping (Phase 7). Both grids
already accept ``labels`` / ``confidences`` in ``insert()``; these helpers just
wire a ``SegmentationResult`` through and build the uniform and foveated maps
from the SAME classified points, so the two are directly comparable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..segmentation.base_segmenter import SegmentationResult
from ..segmentation.classes import NUM_CLASSES
from .foveated_grid import FoveatedGrid
from .resolution_policy import ResolutionPolicy
from .uniform_grid import UniformGrid

logger = logging.getLogger(__name__)


def build_uniform_map(
    points: np.ndarray,
    seg: Optional[SegmentationResult] = None,
    resolution: float = 0.05,
    height_aggregation: str = "max",
    confidence_aggregation: str = "mean",
) -> UniformGrid:
    """Build a uniform-resolution semantic 2.5D map."""
    grid = UniformGrid(
        resolution=resolution,
        height_aggregation=height_aggregation,
        confidence_aggregation=confidence_aggregation,
    )
    labels = seg.labels if seg is not None else None
    conf = seg.confidences if seg is not None else None
    grid.insert(points[:, :3], labels=labels, confidences=conf, n_classes=NUM_CLASSES)
    return grid


def build_foveated_map(
    points: np.ndarray,
    policy: ResolutionPolicy,
    seg: Optional[SegmentationResult] = None,
    height_aggregation: str = "max",
    confidence_aggregation: str = "mean",
) -> FoveatedGrid:
    """Build a foveated (distance-adaptive) semantic 2.5D map."""
    grid = FoveatedGrid(
        policy=policy,
        height_aggregation=height_aggregation,
        confidence_aggregation=confidence_aggregation,
        n_classes=NUM_CLASSES,
    )
    labels = seg.labels if seg is not None else None
    conf = seg.confidences if seg is not None else None
    grid.insert(points[:, :3], labels=labels, confidences=conf)
    return grid


def map_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Pull the height/confidence aggregation choices from config."""
    m = cfg.get("mapping", {})
    return {
        "height_aggregation": m.get("height_aggregation", "max"),
        "confidence_aggregation": m.get("confidence_aggregation", "mean"),
    }
