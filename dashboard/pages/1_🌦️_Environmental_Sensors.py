"""Environmental Sensor Monitoring -- LiDAR data-quality risk (SIH demo).

Separate page from the main pipeline dashboard, as requested: live readings
from the Grove Beginner Kit + the added air quality sensor, classified into
good/moderate/bad against configs/config.yaml -> environmental_sensors, with
an overall verdict on whether current conditions (rain/fog/smoke/dust proxies)
are likely to be degrading the LiDAR point cloud.

Live data path:
    Arduino (arduino/env_lidar_monitor) --USB serial-->
    src/env_sensors/serial_bridge.py (run separately) -->
    data/live/env_sensors.json + env_sensors_history.csv -->
    this page (polls the snapshot on an auto-refreshing fragment)

Run the bridge in its own terminal before opening this page:
    python -m src.env_sensors.serial_bridge --port COM3
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.env_sensors.classifier import classify_snapshot  # noqa: E402
from dashboard.theme import inject_css, STATUS_COLORS, STATUS_LABELS  # noqa: E402

st.set_page_config(
    page_title="Environmental Sensors -- Foveated LiDAR Mapping",
    page_icon="🌦️",
    layout="wide",
)
inject_css()

cfg = load_config()
ENV_CFG = cfg["environmental_sensors"]
THRESHOLDS = ENV_CFG["thresholds"]
SNAPSHOT_PATH = resolve_path(ENV_CFG["live"]["snapshot_path"])
HISTORY_PATH = resolve_path(ENV_CFG["live"]["history_path"])
STALE_AFTER_S = float(ENV_CFG["live"]["stale_after_s"])

# Sensor -> LiDAR-relevant display metadata. Order here is the tile order.
METRICS: List[Dict[str, str]] = [
    {"key": "humidity_pct", "label": "Humidity", "unit": "%RH",
     "why": "High humidity accompanies fog and rain."},
    {"key": "pressure_trend_hpa", "label": "Pressure drop", "unit": "hPa/10min",
     "why": "A fast drop can signal incoming rain or a storm front."},
    {"key": "sound_raw", "label": "Ambient sound", "unit": "raw",
     "why": "Rain on surfaces and wind raise ambient noise."},
    {"key": "vibration_g", "label": "Vibration", "unit": "g",
     "why": "Wind or an unstable mount blurs a LiDAR scan."},
    {"key": "air_quality_raw", "label": "Air quality", "unit": "raw",
     "why": "Smoke, dust and VOCs directly scatter the beam."},
]


# ---------------------------------------------------------------------------
# Data access -- read what the serial bridge last wrote
# ---------------------------------------------------------------------------
def read_snapshot() -> Optional[Dict[str, Any]]:
    if not SNAPSHOT_PATH.is_file():
        return None
    try:
        with SNAPSHOT_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None  # caught mid-write by the bridge -- treat as "no data yet"


def read_history(max_points: int = 300) -> List[Dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-max_points:]


def status_pill(rating: str) -> str:
    return f'<span class="pill pill-{rating}">{STATUS_LABELS[rating]}</span>'


def fmt_value(value: Any, decimals: int = 1) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Environmental Sensor Monitoring")
st.caption("Grove Beginner Kit + air quality sensor → proxy signals for "
           "rain / fog / smoke / dust that can degrade the LiDAR point cloud")


@st.fragment(run_every="2s")
def live_panel() -> None:
    snapshot = read_snapshot()

    if snapshot is None:
        st.warning(
            "No live data yet. Start the serial bridge in a separate terminal:\n\n"
            "`python -m src.env_sensors.serial_bridge --port COM3`",
            icon="⚠️",
        )
        return

    age_s = time.time() - float(snapshot.get("unix_t", 0))
    is_stale = age_s > STALE_AFTER_S

    result = classify_snapshot(snapshot, THRESHOLDS)
    overall = result["overall"]
    per_metric = result["per_metric"]

    # ---- connection status ----
    if is_stale:
        st.error(f"No update in {age_s:.0f}s -- check the serial bridge / USB "
                  f"connection. Showing the last reading received.", icon="\U0001F50C")
    else:
        st.caption(f"\U0001F7E2 Live · updated {age_s:.1f}s ago")

    # ---- overall risk banner ----
    overall_text = {
        "good": "Conditions look clear — low risk of LiDAR interference.",
        "moderate": "Some proxy signals are elevated — watch for degraded returns.",
        "bad": "Conditions likely to distort the point cloud (scattering/attenuation).",
        "unknown": "Not enough sensor data yet to judge conditions.",
    }[overall]
    st.markdown(
        f"""
        <div class="figbox status-{overall}">
          <div class="label">LiDAR data-quality risk · worst reading across all sensors</div>
          <div class="stat" style="color:{STATUS_COLORS[overall]}">{STATUS_LABELS[overall]}</div>
          <div class="sub">{overall_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- per-sensor tiles ----
    cols = st.columns(len(METRICS))
    for col, metric in zip(cols, METRICS):
        rating = per_metric.get(metric["key"], "unknown")
        value_text = fmt_value(snapshot.get(metric["key"]),
                                decimals=0 if metric["unit"] == "raw" else 1)
        col.markdown(
            f"""
            <div class="sensor-tile">
              <div class="name">{metric['label']}</div>
              <div class="value">{value_text}<span class="unit">{metric['unit']}</span></div>
              {status_pill(rating)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        col.caption(metric["why"])

    # ---- raw readings not used in classification, for context ----
    st.markdown("")
    r1, r2 = st.columns(2)
    r1.metric("Temperature", f"{fmt_value(snapshot.get('temp_c'))} °C")
    r2.metric("Air pressure", f"{fmt_value(snapshot.get('pressure_hpa'), 1)} hPa")


live_panel()

# ---------------------------------------------------------------------------
# History trends
# ---------------------------------------------------------------------------
st.header("Recent trend")
history = read_history()

if len(history) < 2:
    st.info("Not enough history yet -- trends appear once the bridge has been "
            "running for a little while.")
else:
    trend_metric = st.selectbox(
        "Metric", [m["key"] for m in METRICS],
        format_func=lambda k: next(m["label"] for m in METRICS if m["key"] == k),
    )
    limit = THRESHOLDS[trend_metric]

    times = [float(r["unix_t"]) for r in history]
    t0 = times[0]
    minutes = [(t - t0) / 60.0 for t in times]
    values = []
    for r in history:
        raw = r.get(trend_metric, "")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(None)

    fig = go.Figure()
    # Status-colored background bands (good/moderate/bad), single-hue line on top.
    y_max = max((v for v in values if v is not None), default=1.0)
    y_min = min((v for v in values if v is not None), default=0.0)
    pad = max((y_max - y_min) * 0.15, 0.5)
    plot_top, plot_bottom = y_max + pad, min(y_min - pad, 0)

    if limit["direction"] == "high_bad":
        bands = [
            (plot_bottom, limit["good_limit"], "good"),
            (limit["good_limit"], limit["moderate_limit"], "moderate"),
            (limit["moderate_limit"], plot_top, "bad"),
        ]
    else:
        bands = [
            (plot_bottom, limit["moderate_limit"], "bad"),
            (limit["moderate_limit"], limit["good_limit"], "moderate"),
            (limit["good_limit"], plot_top, "good"),
        ]
    for lo, hi, rating in bands:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=STATUS_COLORS[rating], opacity=0.08,
                       line_width=0)

    fig.add_trace(go.Scatter(
        x=minutes, y=values, mode="lines", name=trend_metric,
        line=dict(color="#1a1a1a", width=2),
        hovertemplate="%{y:.2f} at t-%{x:.1f} min<extra></extra>",
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Minutes ago (oldest → newest)",
        yaxis_title=trend_metric,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(autorange="reversed", gridcolor="#e2e0db")
    fig.update_yaxes(range=[plot_bottom, plot_top], gridcolor="#e2e0db")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# How this works
# ---------------------------------------------------------------------------
with st.expander("How the risk verdict is computed"):
    st.markdown(
        """
        The Grove Beginner Kit has no dedicated rain/fog/smoke sensor, so this
        page uses five **proxy signals** and takes the *worst* rating among
        them as the overall verdict (a single bad reading is enough to flag
        risk, matching how any one of rain/fog/smoke/dust can independently
        distort a LiDAR return):

        | Sensor | Proxy for | Direction |
        |---|---|---|
        | Humidity (AHT20/DHT20) | Fog / rain likelihood | higher = worse |
        | Pressure trend (BMP280) | Incoming rain/storm | faster drop = worse |
        | Sound level | Rain / wind noise | louder = worse |
        | Vibration (LIS3DHTR) | Wind / mount instability | more = worse |
        | Air quality (Grove AQ v1.3) | Smoke / dust / VOCs | **lower** raw value = worse |

        All limits live in `configs/config.yaml` under `environmental_sensors.thresholds`
        — nothing here is hard-coded. These are honest proxies from a
        low-cost hobby kit, not calibrated meteorological instruments: retune
        the limits against your own deployment site before trusting the
        verdict operationally.
        """
    )

with st.expander("Setup: wiring, Arduino sketch, and the serial bridge"):
    st.markdown(
        f"""
        1. Flash `arduino/env_lidar_monitor/env_lidar_monitor.ino` to the Grove
           Beginner Kit board (Arduino IDE → install the libraries listed
           at the top of the sketch → Upload).
        2. The 0.96" OLED shows every sensor reading at once, plus the
           overall LiDAR-risk verdict on top -- no controls to operate.
        3. Close the Arduino IDE's Serial Monitor (it locks the port), then run
           the bridge from the project root:
           `python -m src.env_sensors.serial_bridge --port COM3`
        4. Leave the bridge running; this page polls
           `{ENV_CFG['live']['snapshot_path']}` every 2 seconds.
        """
    )
