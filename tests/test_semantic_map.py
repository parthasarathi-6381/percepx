"""Phase 7 tests: semantic labels flow from segmentation into 2.5D cells."""
from __future__ import annotations

import numpy as np

from src.mapping import build_foveated_map, build_uniform_map, ResolutionPolicy
from src.segmentation import BaselineSegmenter, CLASS_ID
from src.segmentation.base_segmenter import SegmentationResult


def _policy():
    return ResolutionPolicy(10, 30, 60, 100, 0.05, 0.15, 0.30, 0.50)


def test_uniform_map_carries_labels():
    # Two well-separated cells with distinct classes.
    pts = np.array([[0.1, 0.1, 0.0], [50.1, 50.1, 0.0]])
    seg = SegmentationResult(
        labels=np.array([CLASS_ID["vehicle"], CLASS_ID["pedestrian"]]),
        confidences=np.array([0.9, 0.8]),
        method="test",
    )
    grid = build_uniform_map(pts, seg, resolution=0.05)
    arr = grid.to_arrays()
    classes = set(arr["semantic_class"].tolist())
    assert CLASS_ID["vehicle"] in classes
    assert CLASS_ID["pedestrian"] in classes


def test_foveated_map_carries_labels_across_zones():
    # near (r~1.4) and far (r~63) points with different classes.
    pts = np.array([[1.0, 1.0, 0.0], [45.0, 45.0, 0.0]])  # r = 1.41, 63.6
    seg = SegmentationResult(
        labels=np.array([CLASS_ID["ground"], CLASS_ID["static_obstacle"]]),
        confidences=np.array([0.9, 0.7]),
        method="test",
    )
    grid = build_foveated_map(pts, _policy(), seg)
    arr = grid.to_arrays()
    # one cell in near zone (index 0), one in very_far zone (index 3)
    assert set(arr["zone_index"].tolist()) == {0, 3}
    classes = set(arr["semantic_class"].tolist())
    assert CLASS_ID["ground"] in classes
    assert CLASS_ID["static_obstacle"] in classes


def test_end_to_end_segment_then_map():
    from src.utils.synthetic import generate_frame

    pts = generate_frame(seed=9)
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    pts = pts[r <= 100.0]
    seg = BaselineSegmenter().segment(pts)

    uni = build_uniform_map(pts, seg, resolution=0.05)
    fov = build_foveated_map(pts, _policy(), seg)

    assert uni.num_cells() > 0
    assert fov.num_cells() > 0
    assert fov.num_cells() <= uni.num_cells()
    # ground should be present in both maps
    assert CLASS_ID["ground"] in set(uni.to_arrays()["semantic_class"].tolist())
