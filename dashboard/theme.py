"""Shared "Data-Dense Dashboard" styling for every dashboard page.

Streamlit's multipage model reruns each page's script from scratch, so CSS
injected in one page does not carry over to another -- every page must call
``inject_css()`` itself. Keeping the CSS in one module (instead of copy-pasted
per page) is the single source of truth for the look of the whole dashboard.

Blue/amber analytics palette: deep blue for structure, amber reserved for the
one headline result per page so it actually pops. Status colors (good /
moderate / bad) are reserved for sensor-health state and are never reused as
a categorical series color elsewhere in the dashboard.
"""
from __future__ import annotations

import streamlit as st

STATUS_COLORS = {
    "good": "#059669",
    "moderate": "#D97706",
    "bad": "#DC2626",
    "unknown": "#475569",
}
STATUS_LABELS = {
    "good": "GOOD",
    "moderate": "MODERATE",
    "bad": "BAD",
    "unknown": "NO DATA",
}


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
            --good:      #059669;
            --moderate:  #D97706;
            --bad:       #DC2626;
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
        .figbox.status-good     { border-left-color: var(--good); }
        .figbox.status-moderate { border-left-color: var(--moderate); }
        .figbox.status-bad      { border-left-color: var(--bad); }
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

        /* ---- sensor tile + status pill (env. monitoring page) ---- */
        .sensor-tile {
            border: 1px solid var(--border);
            border-top: 3px solid var(--border-2);
            border-radius: 0 0 8px 8px;
            background: var(--card);
            padding: 0.9rem 1rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(30, 64, 175, 0.05);
        }
        .sensor-tile .name {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--muted);
        }
        .sensor-tile .value {
            font-family: "Fira Code","Consolas",monospace;
            font-size: 1.6rem; font-weight: 600; color: var(--primary);
            margin-top: 0.15rem;
        }
        .sensor-tile .value .unit {
            font-size: 0.95rem; color: var(--muted); font-weight: 500; margin-left: 0.25rem;
        }
        .pill {
            display: inline-block;
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
            padding: 0.15rem 0.55rem; border-radius: 999px;
            margin-top: 0.5rem;
        }
        .pill-good     { background: rgba(5,150,105,0.12);   color: var(--good); }
        .pill-moderate { background: rgba(217,119,6,0.14);   color: var(--moderate); }
        .pill-bad      { background: rgba(220,38,38,0.12);   color: var(--bad); }
        .pill-unknown  { background: rgba(71,85,105,0.12);   color: var(--muted); }

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
