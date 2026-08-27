"""Foveated LiDAR Mapping System -- Streamlit dashboard (SIH demo).

Run:
    streamlit run dashboard/app.py

A judge can: pick a frame -> Run Pipeline -> see raw cloud, semantic cloud,
foveated 2.5D map, uniform-vs-foveated comparison, and performance metrics.
No Python knowledge required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.mapping import ResolutionPolicy  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.utils.synthetic import write_frame  # noqa: E402
from src.visualization import plots  # noqa: E402

st.set_page_config(page_title="Foveated LiDAR Mapping", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_frames() -> list[Path]:
    raw = resolve_path("data/raw")
    if not raw.is_dir():
        return []
    return sorted(raw.glob("*.bin"))


@st.cache_data(show_spinner=False)
def run_pipeline_cached(frame_path: str, overrides: dict, is_synthetic: bool):
    """Run the pipeline and return the result (cached by inputs)."""
    cfg = load_config()
    # Apply sidebar overrides into the config.
    cfg["preprocessing"]["max_range"] = overrides["max_range"]
    cfg["resolution"]["near"] = overrides["res_near"]
    cfg["resolution"]["mid"] = overrides["res_mid"]
    cfg["resolution"]["far"] = overrides["res_far"]
    cfg["resolution"]["very_far"] = overrides["res_very_far"]
    cfg["mapping"]["max_range"] = overrides["max_range"]
    pipeline = Pipeline(cfg)
    result = pipeline.run(frame_path, do_segment=overrides["segment"],
                          is_synthetic=is_synthetic)
    return cfg, result


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")

frames = list_frames()
frame_labels = [f.name for f in frames]

use_synth = st.sidebar.checkbox(
    "Use synthetic frame", value=len(frames) == 0,
    help="Generate a deterministic demo frame (no KITTI dataset required).",
)

selected_frame = None
if not use_synth:
    if frame_labels:
        choice = st.sidebar.selectbox("LiDAR frame (data/raw/*.bin)", frame_labels)
        selected_frame = frames[frame_labels.index(choice)]
    else:
        st.sidebar.info("No .bin frames in data/raw/. Using synthetic frame.")
        use_synth = True

st.sidebar.markdown("### Mapping range")
max_range = st.sidebar.slider("Max range (m)", 20, 120, 100, 5)

st.sidebar.markdown("### Foveated resolutions (m/cell)")
res_near = st.sidebar.number_input("Near (0-10 m)", 0.01, 0.5, 0.05, 0.01, format="%.2f")
res_mid = st.sidebar.number_input("Mid (10-30 m)", 0.05, 1.0, 0.15, 0.05, format="%.2f")
res_far = st.sidebar.number_input("Far (30-60 m)", 0.1, 1.0, 0.30, 0.05, format="%.2f")
res_very_far = st.sidebar.number_input("Very far (60-100 m)", 0.1, 2.0, 0.50, 0.05, format="%.2f")

segment = st.sidebar.checkbox("Run semantic segmentation", value=True,
                              help="Baseline geometric classifier (prototype).")

run = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Foveated LiDAR Mapping System")
st.caption("Distance-adaptive foveated semantic 2.5D representation for "
           "real-time autonomous navigation")

if not run:
    st.info("Set options in the sidebar and click **Run Pipeline** to start.")
    # Show the resolution-zone diagram up front so judges get the concept.
    policy = ResolutionPolicy(10, 30, max_range * 0.3, max_range,
                              res_near, res_mid, res_far, res_very_far)
    st.subheader("Foveated resolution zones")
    st.plotly_chart(plots.plot_resolution_zones(policy), use_container_width=True)
    st.stop()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
overrides = dict(
    max_range=float(max_range), res_near=float(res_near), res_mid=float(res_mid),
    res_far=float(res_far), res_very_far=float(res_very_far), segment=bool(segment),
)

if use_synth:
    frame_path = str(resolve_path("data/raw/synthetic_000000.bin"))
    write_frame(frame_path, seed=42)
    is_syn = True
else:
    frame_path = str(selected_frame)
    is_syn = False

with st.spinner("Running pipeline..."):
    cfg, result = run_pipeline_cached(frame_path, overrides, is_syn)

if is_syn:
    st.warning("Showing a SYNTHETIC demo frame (not real sensor data).")

t = result.timings.as_dict()
b = result.benchmark

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Input points", f"{result.clean_points.shape[0]:,}")
c2.metric("Uniform cells", f"{b.uniform.num_cells:,}")
c3.metric("Foveated cells", f"{b.foveated.num_cells:,}",
          delta=f"-{b.cell_reduction_percent:.1f}%")
c4.metric("Logical memory", f"{b.foveated.logical_bytes/1024:.0f} KB",
          delta=f"-{b.logical_memory_reduction_percent:.1f}%")
c5.metric("Foveated FPS", f"{b.foveated.fps:.1f}")

# ---------------------------------------------------------------------------
# Section 1: RAW LIDAR
# ---------------------------------------------------------------------------
st.header("1 - Raw LiDAR")
st.plotly_chart(
    plots.plot_raw_pointcloud(
        result.clean_points,
        max_points=int(cfg["visualization"]["max_points_render"]),
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Section 2: SEMANTIC LIDAR
# ---------------------------------------------------------------------------
st.header("2 - Semantic LiDAR")
if result.segmentation is not None:
    if result.segmentation.is_prototype:
        st.info("Labels from a PROTOTYPE geometric classifier "
                "(not a trained neural network).")
    st.plotly_chart(
        plots.plot_semantic_pointcloud(
            result.clean_points, result.segmentation.labels, cfg,
            max_points=int(cfg["visualization"]["max_points_render"]),
        ),
        use_container_width=True,
    )
else:
    st.info("Segmentation was disabled.")

# ---------------------------------------------------------------------------
# Section 3: FOVEATED 2.5D MAP
# ---------------------------------------------------------------------------
st.header("3 - Foveated 2.5D Map")
color_by = st.radio("Color cells by", ["elevation", "semantic"],
                    horizontal=True, key="fov_color")
st.plotly_chart(
    plots.plot_grid_map(result.foveated_grid.to_arrays(), cfg,
                        color_by=color_by, title="Foveated 2.5D Map"),
    use_container_width=True,
)
zone_cols = st.columns(4)
for col, (zname, z) in zip(zone_cols, b.per_zone.items()):
    col.metric(f"{zname} @ {z['resolution']:g} m", f"{z['num_cells']:,} cells",
               help=f"{z['points']:,} points")

# ---------------------------------------------------------------------------
# Section 4: UNIFORM vs FOVEATED
# ---------------------------------------------------------------------------
st.header("4 - Uniform vs Foveated")
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(
        plots.plot_grid_map(result.uniform_grid.to_arrays(), cfg,
                            color_by="elevation", title="Uniform 2.5D (5 cm)"),
        use_container_width=True,
    )
with cc2:
    st.plotly_chart(
        plots.plot_grid_map(result.foveated_grid.to_arrays(), cfg,
                            color_by="elevation", title="Foveated 2.5D"),
        use_container_width=True,
    )
st.plotly_chart(plots.plot_comparison_bars(b), use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: PERFORMANCE
# ---------------------------------------------------------------------------
st.header("5 - Performance")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Cell reduction", f"{b.cell_reduction_percent:.1f}%")
p2.metric("Memory reduction", f"{b.logical_memory_reduction_percent:.1f}%")
p3.metric("Uniform mapping", f"{b.uniform.mapping_time_s*1e3:.1f} ms")
p4.metric("Foveated mapping", f"{b.foveated.mapping_time_s*1e3:.1f} ms")

with st.expander("Per-stage timing"):
    st.table({k: [f"{v*1e3:.2f} ms"] for k, v in t.items()})

with st.expander("Preprocessing report (points discarded)"):
    st.json(result.preprocess_stats.as_dict())

with st.expander("Benchmark notes (how numbers are measured)"):
    for note in b.notes:
        st.write("- " + note)

st.caption("All metrics are real measurements on the selected frame. "
           "Foveation's main win is fewer cells -> less memory; mapping "
           "latency is comparable between uniform and foveated (both real-time).")
