"""Semantic segmentation subpackage.

Public API:
    BaseSegmenter, SegmentationResult   -- the common interface
    BaselineSegmenter                   -- geometric prototype (default)
    PretrainedSegmenter                 -- deep-learning stub (interface only)
    build_segmenter(cfg)                -- factory keyed on config
    ClassRegistry, CLASS_NAMES, ...     -- class id/color registry
"""

from .base_segmenter import BaseSegmenter, SegmentationResult
from .baseline_segmenter import BaselineConfig, BaselineSegmenter
from .pretrained_segmenter import PretrainedSegmenter
from .factory import build_segmenter
from .classes import (
    CLASS_ID,
    CLASS_NAMES,
    DEFAULT_COLORS,
    NUM_CLASSES,
    ClassRegistry,
)

__all__ = [
    "BaseSegmenter",
    "SegmentationResult",
    "BaselineSegmenter",
    "BaselineConfig",
    "PretrainedSegmenter",
    "build_segmenter",
    "ClassRegistry",
    "CLASS_ID",
    "CLASS_NAMES",
    "DEFAULT_COLORS",
    "NUM_CLASSES",
]
