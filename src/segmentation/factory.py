"""Segmenter factory: pick the segmenter from config without the pipeline
knowing the concrete classes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from .base_segmenter import BaseSegmenter
from .baseline_segmenter import BaselineConfig, BaselineSegmenter
from .pretrained_segmenter import PretrainedSegmenter

logger = logging.getLogger(__name__)


def build_segmenter(cfg: Dict[str, Any]) -> BaseSegmenter:
    """Return a segmenter instance based on ``segmentation.method`` in config."""
    method = cfg.get("segmentation", {}).get("method", "baseline").lower()

    if method == "baseline":
        return BaselineSegmenter(BaselineConfig.from_config(cfg))
    if method == "pretrained":
        logger.info("Config requests pretrained segmenter (stub).")
        return PretrainedSegmenter.from_config(cfg)

    raise ValueError(
        f"Unknown segmentation.method: {method!r}. Use 'baseline' or 'pretrained'."
    )
