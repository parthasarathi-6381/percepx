"""Plotly-based visualizations for the foveated LiDAR pipeline.

Plotly is used as the default backend so the MVP has no hard dependency on
Open3D. Functions here return Plotly ``Figure`` objects; the caller decides
whether to ``.show()`` them, write HTML, or embed them in Streamlit.

Provides: raw cloud, semantic cloud, uniform/foveated 2.5D maps, resolution
zones, and a uniform-vs-foveated comparison bar chart.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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

# Dashboard design tokens -- "Data-Dense Dashboard" (blue/amber analytics).
_INK = "#1E3A8A"                 # deep blue text / titles
_MUTED = "#475569"               # axis labels
_GRID = "rgba(30,64,175,0.08)"   # faint blue-tinted grid
_PANEL = "rgba(0,0,0,0)"         # transparent so it sits on the card background
_ACCENT = "#D97706"              # amber -- reserved for the headline series


def _apply_theme(fig):
    """Apply the blue/amber dashboard template (light card, blue-tinted grid)."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=_PANEL,
        plot_bgcolor=_PANEL,
        font=dict(color=_INK, size=13,
                  family="Fira Sans, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"),
        title_font=dict(color=_INK, size=15, family="Fira Sans, -apple-system, Segoe UI, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED)),
        colorway=["#1E40AF", "#D97706", "#3B82F6", "#059669", "#7C3AED", "#DC2626"],
    )
    # 2D axes (ignored harmlessly by 3D scenes)
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID,
                     linecolor="#BFDBFE", tickfont=dict(color=_MUTED,
                     family="Fira Code, Consolas, monospace"))
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID,
                     linecolor="#BFDBFE", tickfont=dict(color=_MUTED,
                     family="Fira Code, Consolas, monospace"))
    return fig


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
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# Class metadata helpers
# ---------------------------------------------------------------------------
def class_names_colors(cfg: Optional[Dict[str, Any]] = None):
    """Return (names, colors) lists ordered by class id, from config or defaults."""
    from src.segmentation.classes import CLASS_NAMES, DEFAULT_COLORS

    if not cfg or "classes" not in cfg:
        names = list(CLASS_NAMES)
        return names, [DEFAULT_COLORS[n] for n in names]
    items = sorted(cfg["classes"].items(), key=lambda kv: kv[1].get("id", 0))
    names = [n for n, _ in items]
    colors = [m.get("color", DEFAULT_COLORS.get(n, "#7f7f7f")) for n, m in items]
    return names, colors


# ---------------------------------------------------------------------------
# Semantic point cloud
# ---------------------------------------------------------------------------
def plot_semantic_pointcloud(
    points: np.ndarray,
    labels: np.ndarray,
    cfg: Optional[Dict[str, Any]] = None,
    max_points: int = 120_000,
    point_size: float = 1.6,
    title: str = "Semantic LiDAR Point Cloud",
):
    """3D scatter colored by semantic class (one trace per class -> legend)."""
    import plotly.graph_objects as go

    names, colors = class_names_colors(cfg)
    idx = _subsample(points, max_points)
    p = points[idx]
    lab = np.asarray(labels)[idx]

    traces = []
    for cid, (name, color) in enumerate(zip(names, colors)):
        m = lab == cid
        if not m.any():
            continue
        traces.append(
            go.Scatter3d(
                x=p[m, 0], y=p[m, 1], z=p[m, 2],
                mode="markers", name=name,
                marker=dict(size=point_size, color=color, opacity=0.8),
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title}  ({len(idx):,} pts)",
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 2.5D grid map (top-down, colored by elevation or by semantic class)
# ---------------------------------------------------------------------------
def plot_grid_map(
    arrays: Dict[str, np.ndarray],
    cfg: Optional[Dict[str, Any]] = None,
    color_by: str = "elevation",
    title: str = "2.5D Map",
    marker_size: float = 3.0,
):
    """Top-down scatter of occupied cells.

    ``arrays`` is the dict from ``UniformGrid.to_arrays`` /
    ``FoveatedGrid.to_arrays``. ``color_by`` = 'elevation' | 'semantic'.
    """
    import plotly.graph_objects as go

    x, y = arrays["x"], arrays["y"]
    if x.shape[0] == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (empty)")
        return _apply_theme(fig)

    if color_by == "semantic":
        names, colors = class_names_colors(cfg)
        cls = arrays["semantic_class"]
        traces = []
        for cid, (name, color) in enumerate(zip(names, colors)):
            m = cls == cid
            if not m.any():
                continue
            traces.append(go.Scattergl(
                x=x[m], y=y[m], mode="markers", name=name,
                marker=dict(size=marker_size, color=color)))
        fig = go.Figure(data=traces)
    else:
        fig = go.Figure(data=[go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=marker_size, color=arrays["elevation"],
                        colorscale="Turbo", colorbar=dict(title="elev (m)")),
        )])

    fig.update_layout(
        title=f"{title}  ({x.shape[0]:,} cells)",
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# Resolution zones (rings showing where each cell size applies)
# ---------------------------------------------------------------------------
def plot_resolution_zones(policy, title: str = "Foveated resolution zones"):
    """Filled concentric bands illustrating distance-adaptive resolution.

    Blue/amber palette: the innermost (highest-resolution) zone is amber to
    draw the eye to the "fovea", outer zones fade through blue -- so the
    adaptive-resolution idea reads at a glance.
    """
    import plotly.graph_objects as go

    theta = np.linspace(0, 2 * np.pi, 240)
    # Near = amber (the fovea, highest detail); farther zones fade to blue.
    fills = ["rgba(217,119,6,0.55)", "rgba(30,64,175,0.45)",
             "rgba(30,64,175,0.25)", "rgba(30,64,175,0.10)"]
    fig = go.Figure()
    # Draw largest ring first so inner (denser) rings paint on top.
    for zi in range(len(policy.zones) - 1, -1, -1):
        z = policy.zones[zi]
        fig.add_trace(go.Scatter(
            x=z.r_hi * np.cos(theta), y=z.r_hi * np.sin(theta),
            mode="lines", fill="toself",
            fillcolor=fills[zi % len(fills)],
            line=dict(color="rgba(30,64,175,0.9)", width=1),
            name=f"{z.name} · {z.r_lo:g}–{z.r_hi:g} m · {z.resolution*100:g} cm",
            hoverinfo="name",
        ))
    # Sensor / vehicle marker at the origin.
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(color="#ffffff", size=9, line=dict(color="#1E40AF", width=2)),
        text=["vehicle"], textposition="top center",
        textfont=dict(color="#1E3A8A", size=11), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="top", y=-0.28, x=0.5, xanchor="center",
                    entrywidth=260, entrywidthmode="pixels"),
        margin=dict(l=0, r=0, t=40, b=110),
    )
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# Uniform vs foveated comparison bars
# ---------------------------------------------------------------------------
def plot_comparison_bars(benchmark, title: str = "Uniform vs Foveated"):
    """Grouped bars: cells, logical KB, FPS for uniform vs foveated."""
    import plotly.graph_objects as go

    u, f = benchmark.uniform, benchmark.foveated
    metrics = ["cells", "logical KB", "FPS"]
    uni_vals = [u.num_cells, u.logical_bytes / 1024, u.fps]
    fov_vals = [f.num_cells, f.logical_bytes / 1024, f.fps]

    # Separate subplots would be cleaner (different scales); use a normalized
    # grouped bar per metric via facets. Keep it simple: 3 small bar charts.
    from plotly.subplots import make_subplots

    # Blue/amber palette: uniform = light blue (baseline), foveated = amber (the win).
    uni_color, fov_color = "#93C5FD", "#D97706"
    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
    for i, (uv, fv) in enumerate(zip(uni_vals, fov_vals), start=1):
        fig.add_trace(go.Bar(x=["uniform"], y=[uv], marker_color=uni_color,
                             showlegend=(i == 1), name="uniform"), row=1, col=i)
        fig.add_trace(go.Bar(x=["foveated"], y=[fv], marker_color=fov_color,
                             showlegend=(i == 1), name="foveated"), row=1, col=i)
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=60, b=0),
                      bargap=0.35)
    return _apply_theme(fig)
