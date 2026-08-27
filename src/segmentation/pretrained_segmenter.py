"""Pretrained deep-learning segmenter -- INTERFACE / STUB only.

This is where a real point-cloud network (PointNet++, a sparse-convolution model
like MinkowskiNet / SPVNAS, RandLA-Net, ...) would be plugged in. It deliberately
does NOT fake predictions: if no model/weights are configured, ``segment`` raises
a clear error explaining what is required, rather than returning made-up labels.

Because it implements the same ``BaseSegmenter`` interface and returns a
``SegmentationResult``, dropping in a trained model requires **no changes to the
mapping engine** -- only filling in ``_load_model`` and ``_predict`` below.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base_segmenter import BaseSegmenter, SegmentationResult

logger = logging.getLogger(__name__)


class PretrainedSegmenter(BaseSegmenter):
    """Stub for a trained point-cloud segmentation model."""

    name = "pretrained"
    is_prototype = False  # a real trained model would NOT be a prototype

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.weights_path = weights_path
        self.device = self._resolve_device(device)
        self.config = config or {}
        self._model = None  # loaded lazily

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PretrainedSegmenter":
        p = cfg.get("segmentation", {}).get("pretrained", {})
        return cls(
            weights_path=p.get("weights_path"),
            device=p.get("device", "auto"),
            config=p,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Detect GPU if requested 'auto', else honor explicit choice.

        Never hard-fails: if torch is missing, returns 'cpu' and the real error
        surfaces only when someone actually tries to run the (unimplemented)
        model.
        """
        if device != "auto":
            return device
        try:
            import torch  # noqa: F401

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load weights. NOT IMPLEMENTED in the MVP."""
        raise NotImplementedError(
            "PretrainedSegmenter is a stub. To use a deep-learning model:\n"
            "  1. pip install torch (and the model's package)\n"
            "  2. set segmentation.method: pretrained and a valid\n"
            "     segmentation.pretrained.weights_path in config.yaml\n"
            "  3. implement _load_model() and _predict() here.\n"
            "Until then, use the BaselineSegmenter (segmentation.method: baseline)."
        )

    def _predict(self, points: np.ndarray) -> SegmentationResult:  # pragma: no cover
        """Run inference. NOT IMPLEMENTED in the MVP."""
        raise NotImplementedError("PretrainedSegmenter._predict is not implemented.")

    # ------------------------------------------------------------------
    def segment(self, points: np.ndarray) -> SegmentationResult:
        weights_ok = self.weights_path and Path(self.weights_path).is_file()
        if not weights_ok:
            raise NotImplementedError(
                "No pretrained weights found at "
                f"{self.weights_path!r}. The pretrained segmenter is a stub; "
                "use the baseline segmenter or provide a trained model. "
                "(See PretrainedSegmenter docstring.)"
            )
        # If weights existed, this is where loading + inference would run:
        if self._model is None:
            self._load_model()
        return self._predict(points)
