"""Common segmenter interface.

Every segmenter (baseline geometric, future PointNet++/sparse-CNN, ...) returns
the SAME structured result, so the mapping engine never has to know which model
produced the labels. This is the seam that lets us swap the baseline for a deep
model later without touching the grids.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SegmentationResult:
    """Per-point semantic labels + confidences for one frame.

    Attributes
    ----------
    labels:
        ``(N,)`` int array of class ids (indices into the class registry).
    confidences:
        ``(N,)`` float array in [0, 1].
    method:
        Name of the segmenter that produced this (for provenance / dashboard).
    is_prototype:
        True if these labels come from a heuristic prototype (NOT a trained
        network). The dashboard shows this so results are never oversold.
    meta:
        Optional extra info (e.g. #points per class, ground plane params).
    """

    labels: np.ndarray
    confidences: np.ndarray
    method: str
    is_prototype: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels).astype(np.int64)
        self.confidences = np.asarray(self.confidences).astype(np.float64)
        if self.labels.shape != self.confidences.shape:
            raise ValueError(
                "labels and confidences must have the same shape: "
                f"{self.labels.shape} vs {self.confidences.shape}"
            )

    @property
    def num_points(self) -> int:
        return int(self.labels.shape[0])


class BaseSegmenter(abc.ABC):
    """Abstract base class for all point-cloud segmenters."""

    #: Human-readable name (subclasses override).
    name: str = "base"

    #: Whether this segmenter is a heuristic prototype (True) or a trained
    #: model (False). Subclasses MUST set this honestly.
    is_prototype: bool = True

    @abc.abstractmethod
    def segment(self, points: np.ndarray) -> SegmentationResult:
        """Classify each point.

        Parameters
        ----------
        points:
            ``(N, >=3)`` array; columns 0,1,2 are x,y,z. A 4th intensity column
            may be present and may or may not be used.

        Returns
        -------
        SegmentationResult
        """
        raise NotImplementedError
