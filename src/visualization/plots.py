"""Plotly-based visualizations for the foveated LiDAR pipeline.

Plotly is used as the default backend so the MVP has no hard dependency on
Open3D. Functions here return Plotly ``Figure`` objects; the caller decides
whether to ``.show()`` them, write HTML, or embed them in Streamlit.

Only the raw point-cloud view is implemented in Phase 2. Semantic / grid
visualizations are added in later phases.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Semantic class -> color. Kept in sync with configs/config.yaml.
CLASS_COLORS = {
    "ground": "#2ca02c",          # green
    "vehicle": "#d62728",         # red
    "pedestrian": "#ffdd00",      # yellow
    "static_obstacle": "#1f77b4",  # blue
    "unknown": "#7f7f7f",         # gray
}


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Return an index array subsampling ``points`` to at most ``max_points``.

    Deterministic (evenly spaced stride) so repeated renders look identical.
    """
    n = points.shape[0]
    if n <= max_points or max_points <= 0:
        return np.arange(n)
    step = int(np.ceil(n / max_points))
    return np.arange(0, n, step)


def plot_raw_pointcloud(
    points: np.ndarray,
    max_points: int = 120_000,
    point_size: float = 1.5,
    title: str = "Raw LiDAR Point Cloud",
):
    """Return a 3D scatter Figure of a raw ``(N, 4)`` point cloud.

    Points are colored by intensity (4th column) so structure is visible even
    before semantic segmentation.
    """
    import plotly.graph_objects as go  # imported lazily so tests don't need it

    idx = _subsample(points, max_points)
    p = points[idx]
    intensity = p[:, 3] if p.shape[1] > 3 else np.zeros(p.shape[0])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=intensity,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="intensity"),
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"{title}  (showing {len(idx):,} / {points.shape[0]:,} points)",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
