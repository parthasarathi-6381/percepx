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

st.set_page_config(
    page_title="Foveated LiDAR Mapping",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Design system -- "Data-Dense Dashboard" (blue/amber analytics theme).
# Light slate surfaces, deep-blue ink for structure, amber reserved for the
# one headline result so it pops against an otherwise calm, technical UI.
# Fira Sans for text, Fira Code for every number so metrics stay tabular.
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg:        #F8FAFC;   /* app background, cool light slate      */
            --card:      #FFFFFF;   /* panels / metric tiles                 */
            --primary:   #1E40AF;   /* deep blue -- headings, structure      */
            --primary-2: #3B82F6;   /* lighter blue -- links, secondary emph */
            --accent:    #D97706;   /* amber -- the ONE headline result      */
            --accent-bg: #FFFBEB;   /* warm amber tint for the hero panel    */
            --ink:       #1E3A8A;   /* headings                              */
            --text:      #0F172A;   /* body text                             */
            --muted:     #475569;   /* secondary labels                      */
            --border:    #DBEAFE;   /* blue-tinted hairline                  */
            --border-2:  #BFDBFE;   /* stronger border                       */
        }

        html, body, [class*="css"],
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
        button, input, select, textarea, table, th, td,
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stCaption, label, .stRadio, .stSelectbox, .stSlider, .stCheckbox,
        .stAlert, [data-testid="stExpander"], [data-testid="stTable"],
        [data-testid="stDataFrame"], [data-testid="stJson"] {
            font-family: "Fira Sans", -apple-system, "Segoe UI", Roboto, sans-serif !important;
        }
        html, body, [class*="css"] { color: var(--text); }
        .block-container { padding-top: 2rem; max-width: 1220px; }

        /* numeric widgets (sliders, number inputs, dataframes, json) read as data -> mono */
        input[type="number"], [data-testid="stSlider"] [data-testid="stTickBar"],
        [data-testid="stSlider"] div[role="slider"],
        [data-testid="stDataFrame"] *, [data-testid="stJson"] *, .stTable table {
            font-family: "Fira Code", "Consolas", monospace !important;
        }

        /* ---- typography: sans headings, mono numbers ---- */
        h1, h2, h3, h4 {
            font-family: "Fira Sans", -apple-system, "Segoe UI", sans-serif;
            color: var(--ink);
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        h1 { font-size: 2.15rem; line-height: 1.15; }

        /* ---- title rule: solid blue heading with a clean underline ---- */
        h1 {
            padding-bottom: 0.6rem;
            border-bottom: 3px solid var(--primary);
            color: var(--primary);
            display: inline-block;
        }

        /* ---- section headers: bold label + accent-colored index number ---- */
        h2 {
            font-size: 1.2rem;
            margin-top: 2.4rem;
            padding: 0.5rem 0.9rem;
            background: var(--card);
            border-left: 4px solid var(--primary);
            border-radius: 0 6px 6px 0;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        /* ---- metrics: elevated cards, blue labels, mono values ---- */
        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 3px rgba(30, 64, 175, 0.06);
        }
        [data-testid="stMetricValue"] {
            font-family: "Fira Code", "Consolas", monospace;
            font-weight: 600;
            color: var(--primary);
            font-size: 1.5rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        [data-testid="stMetricDelta"] { font-family: "Fira Code","Consolas",monospace; }

        /* ---- primary button: solid blue, amber on hover for energy ---- */
        .stButton > button[kind="primary"] {
            background: var(--primary);
            color: #ffffff;
            border: 1px solid var(--primary);
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: 0 1px 3px rgba(30, 64, 175, 0.25);
            transition: background 150ms ease, box-shadow 150ms ease;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--accent);
            border-color: var(--accent);
            box-shadow: 0 2px 8px rgba(217, 119, 6, 0.35);
        }

        /* ---- headline result: amber-tinted hero, the one "wow" moment ---- */
        .figbox {
            border: 1px solid #FDE68A;
            border-left: 5px solid var(--accent);
            background: var(--accent-bg);
            border-radius: 0 10px 10px 0;
            padding: 1.1rem 1.4rem;
            margin: 0.6rem 0 1.6rem 0;
            box-shadow: 0 2px 10px rgba(217, 119, 6, 0.10);
        }
        .figbox .label {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: #92400E; font-weight: 600;
        }
        .figbox .stat {
            font-family: "Fira Code","Consolas",monospace;
            font-size: 2.1rem; font-weight: 700; color: #92400E;
            line-height: 1.25; margin-top: 0.25rem;
        }
        .figbox .sub {
            color: #78716C; font-size: 0.92rem; margin-top: 0.2rem;
        }

        /* ---- landing stat cards (outcome at a glance, no prose) ---- */
        .statcard {
            border: 1px solid var(--border);
            border-top: 4px solid var(--primary);
            border-radius: 0 0 10px 10px;
            background: var(--card);
            padding: 1.2rem 1.3rem 1.1rem;
            height: 100%;
            box-shadow: 0 1px 4px rgba(30, 64, 175, 0.06);
        }
        .statcard .num {
            font-family: "Fira Code","Consolas",monospace;
            font-size: 2.3rem; font-weight: 700; color: var(--primary); line-height: 1;
        }
        .statcard .cap {
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--muted); margin-top: 0.55rem; font-weight: 600;
        }

        /* ---- compact zone table ---- */
        table.zones { width: 100%; border-collapse: collapse; margin-top: 0.2rem; }
        table.zones th, table.zones td {
            text-align: left; padding: 0.55rem 0.5rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }
        table.zones th {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--muted); border-bottom: 2px solid var(--border-2);
        }
        table.zones td.cell {
            font-family: "Fira Code","Consolas",monospace; text-align: right;
            font-weight: 600; color: var(--primary);
        }
        table.zones td.dist { color: var(--muted); }

        /* ---- plotly charts: card container with a soft shadow ---- */
        [data-testid="stPlotlyChart"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem;
            box-shadow: 0 1px 4px rgba(30, 64, 175, 0.05);
        }

        /* ---- expanders: card styling to match metrics/charts ---- */
        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--card);
        }

        /* ---- sidebar: deep blue-tinted panel ---- */
        [data-testid="stSidebar"] {
            background: #EFF6FF;
            border-right: 1px solid var(--border-2);
        }
        /* --- trim the large default whitespace at the top of the sidebar --- */
        [data-testid="stSidebarHeader"] {
            padding-top: 0.25rem !important;
            padding-bottom: 0 !important;
            height: auto !important;
            min-height: 0 !important;
        }
        [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0 !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.25rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:empty {
            display: none !important;
        }
        /* compact sidebar title + section subheads */
        .side-title {
            font-size: 1.3rem; font-weight: 700; color: var(--ink);
            padding-bottom: 0.4rem; margin-bottom: 0.7rem;
            border-bottom: 3px solid var(--primary);
        }
        [data-testid="stSidebar"] h3 {
            font-size: 0.95rem; margin-top: 1.2rem; margin-bottom: 0.3rem;
            color: var(--primary);
        }

        /* mono for inline "frame name" style code */
        code {
            font-family: "Fira Code","Consolas",monospace;
            background: var(--border); color: var(--ink);
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


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
st.sidebar.markdown(
    '<div class="side-title">Controls</div>', unsafe_allow_html=True)

frames = list_frames()
frame_labels = [f.name for f in frames]

# Use the real KITTI frames in data/raw/. If none exist, fall back silently to
# a generated synthetic frame (keeps the demo working with no dataset).
selected_frame = None
if frame_labels:
    use_synth = False
    choice = st.sidebar.selectbox("LiDAR frame (data/raw/*.bin)", frame_labels)
    selected_frame = frames[frame_labels.index(choice)]
else:
    use_synth = True
    st.sidebar.info("No .bin frames in data/raw/ — using a synthetic demo frame.")

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
st.title("Foveated LiDAR Mapping")
st.caption("Distance-adaptive foveated semantic 2.5D representation for "
           "real-time autonomous navigation")

if not run:
    # Outcome at a glance -- three stat cards, no prose.
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        '<div class="statcard"><div class="num">~39%</div>'
        '<div class="cap">Fewer map cells</div></div>', unsafe_allow_html=True)
    s2.markdown(
        '<div class="statcard"><div class="num">~39%</div>'
        '<div class="cap">Less map memory</div></div>', unsafe_allow_html=True)
    s3.markdown(
        '<div class="statcard"><div class="num">2.5D</div>'
        '<div class="cap">Height preserved</div></div>', unsafe_allow_html=True)

    st.markdown("")  # spacer

    # Zone diagram + compact zone table, side by side (visual, not paragraphs).
    left, right = st.columns([1.5, 1])
    with left:
        policy = ResolutionPolicy(10, 30, max_range * 0.6, max_range,
                                  res_near, res_mid, res_far, res_very_far)
        st.plotly_chart(plots.plot_resolution_zones(policy),
                        use_container_width=True)
    with right:
        st.markdown("##### Resolution zones")
        st.markdown(
            f"""
            <table class="zones">
              <tr><th>Zone</th><th>Distance</th><th style="text-align:right">Cell</th></tr>
              <tr><td>Near</td><td class="dist">0–10 m</td><td class="cell">{res_near*100:g} cm</td></tr>
              <tr><td>Mid</td><td class="dist">10–30 m</td><td class="cell">{res_mid*100:g} cm</td></tr>
              <tr><td>Far</td><td class="dist">30–60 m</td><td class="cell">{res_far*100:g} cm</td></tr>
              <tr><td>Very far</td><td class="dist">60–100 m</td><td class="cell">{res_very_far*100:g} cm</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Select a frame in the sidebar → **Run Pipeline**.")
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
    st.warning("Showing a SYNTHETIC demo frame (not real sensor data). "
               "Add KITTI .bin frames to data/raw/ to use real data.")
else:
    st.success(f"Real LiDAR frame: **{Path(frame_path).name}**")

t = result.timings.as_dict()
b = result.benchmark

# ---------------------------------------------------------------------------
# Hero banner: the headline reduction
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="figbox">
      <div class="label">Key result &middot; foveated vs. uniform 5&nbsp;cm grid, same frame</div>
      <div class="stat">&minus;{b.cell_reduction_percent:.1f}% cells &nbsp;&middot;&nbsp;
        &minus;{b.logical_memory_reduction_percent:.1f}% map memory</div>
      <div class="sub">{b.uniform.num_cells:,} &rarr; {b.foveated.num_cells:,} occupied cells,
        with full 5&nbsp;cm detail retained in the near field.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
p3.metric("Uniform mapping", f"{b.uniform.mapping_time_s*1e3:.1f} ms",
          help=f"{b.uniform.fps:.0f} FPS")
p4.metric("Foveated mapping", f"{b.foveated.mapping_time_s*1e3:.1f} ms",
          help=f"{b.foveated.fps:.0f} FPS")

# Per-stage timing at a glance (segmentation is the prototype-classifier cost).
q1, q2, q3, q4 = st.columns(4)
q1.metric("Preprocess", f"{t['preprocessing_time']*1e3:.0f} ms")
q2.metric("Segmentation", f"{t['segmentation_time']*1e3:.0f} ms",
          help="Baseline prototype classifier (RANSAC ground + voxel DBSCAN).")
q3.metric("Mapping (both)",
          f"{(t['uniform_mapping_time']+t['foveated_mapping_time'])*1e3:.0f} ms")
q4.metric("Total", f"{t['total_time']*1e3:.0f} ms")

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
