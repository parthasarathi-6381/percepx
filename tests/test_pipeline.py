"""Integration tests for the central pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.pipeline import Pipeline
from src.utils.synthetic import write_frame


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "frame.bin"
    write_frame(str(p), seed=42)
    return str(p)


def test_pipeline_runs_end_to_end(cfg, frame):
    result = Pipeline(cfg).run(frame, is_synthetic=True)
    assert result.raw_points.shape[0] > 0
    assert result.clean_points.shape[0] > 0
    assert result.uniform_grid.num_cells() > 0
    assert result.foveated_grid.num_cells() > 0
    # foveated has <= uniform cells
    assert result.foveated_grid.num_cells() <= result.uniform_grid.num_cells()


def test_pipeline_timings_present_and_positive(cfg, frame):
    result = Pipeline(cfg).run(frame, is_synthetic=True)
    t = result.timings.as_dict()
    for key in ("load_time", "preprocessing_time", "uniform_mapping_time",
                "foveated_mapping_time", "total_time"):
        assert t[key] >= 0
    # total should be >= sum of major stages (it wraps them)
    assert t["total_time"] >= t["uniform_mapping_time"]


def test_pipeline_benchmark_consistent(cfg, frame):
    result = Pipeline(cfg).run(frame, is_synthetic=True)
    b = result.benchmark
    assert b.num_points == result.clean_points.shape[0]
    assert b.uniform.num_cells == result.uniform_grid.num_cells()
    assert b.foveated.num_cells == result.foveated_grid.num_cells()


def test_pipeline_no_segment(cfg, frame):
    result = Pipeline(cfg).run(frame, do_segment=False, is_synthetic=True)
    assert result.segmentation is None
    assert result.timings.segmentation_time == 0.0
    # maps still built
    assert result.uniform_grid.num_cells() > 0


def test_pipeline_summary_json_serializable(cfg, frame):
    import json

    result = Pipeline(cfg).run(frame, is_synthetic=True)
    s = result.summary_dict()
    # must be JSON serializable (default=str for any numpy leftovers)
    txt = json.dumps(s, default=str)
    assert "timings" in txt and "benchmark" in txt


def test_pipeline_segmentation_flagged_prototype(cfg, frame):
    result = Pipeline(cfg).run(frame, is_synthetic=True)
    assert result.segmentation is not None
    assert result.segmentation.is_prototype is True
