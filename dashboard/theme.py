"""Shared "Clean light / report" styling for every dashboard page.

Streamlit's multipage model reruns each page's script from scratch, so CSS
injected in one page does not carry over to another -- every page must call
``inject_css()`` itself. Keeping the CSS in one module (instead of copy-pasted
per page) is the single source of truth for the look of the whole dashboard.
"""
from __future__ import annotations

import streamlit as st

# Status colors are reserved for good/moderate/bad state and are never reused
# as a categorical series color elsewhere in the dashboard.
STATUS_COLORS = {
    "good": "#1a7f37",
    "moderate": "#b58105",
    "bad": "#c92a2a",
    "unknown": "#6b6b6b",
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
        :root {
            --paper:   #ffffff;
            --panel:   #faf9f7;   /* faint warm gray for slight separation */
            --ink:     #1a1a1a;   /* near-black body text                  */
            --muted:   #6b6b6b;   /* secondary labels                      */
            --rule:    #e2e0db;   /* hairline dividers / borders           */
            --rule-2:  #cfccc5;   /* slightly stronger rule                */
            --accent:  #1a1a1a;   /* accent = ink (Swiss restraint)        */
            --good:    #1a7f37;
            --moderate:#b58105;
            --bad:     #c92a2a;
        }

        html, body, [class*="css"] { color: var(--ink); }
        .block-container { padding-top: 2.4rem; max-width: 1180px; }

        /* ---- typography: serif display headings, sans body, mono numbers ---- */
        h1, h2, h3, h4 {
            font-family: Georgia, "Times New Roman", serif;
            color: var(--ink);
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        h1 { font-size: 2.1rem; line-height: 1.15; }
        .stMarkdown p, .stMarkdown li, label, .stCaption {
            font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* ---- title rule: a thin line under the H1, like a paper masthead ---- */
        h1 { padding-bottom: 0.5rem; border-bottom: 2px solid var(--ink); }

        /* ---- section headers: small-caps label + hairline rule ---- */
        h2 {
            font-size: 1.15rem;
            text-transform: none;
            margin-top: 2.2rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid var(--rule-2);
        }

        /* ---- metrics: flat, ruled tiles; VALUES in monospace ---- */
        [data-testid="stMetric"] {
            background: var(--paper);
            border: none;
            border-top: 1px solid var(--rule);
            border-radius: 0;
            padding: 0.7rem 0.9rem 0.7rem 0;
        }
        [data-testid="stMetricValue"] {
            font-family: "SF Mono", "Consolas", "Liberation Mono", monospace;
            font-weight: 600;
            color: var(--ink);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricDelta"] { font-family: "SF Mono","Consolas",monospace; }

        /* ---- primary button: solid ink, square, no gradient/shadow ---- */
        .stButton > button[kind="primary"] {
            background: var(--ink);
            color: var(--paper);
            border: 1px solid var(--ink);
            border-radius: 3px;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: none;
        }
        .stButton > button[kind="primary"]:hover {
            background: #000; color: #fff;
        }

        /* ---- headline result: a bordered "figure box", not a gradient hero -- */
        .figbox {
            border: 1px solid var(--rule-2);
            border-left: 3px solid var(--ink);
            background: var(--panel);
            padding: 1rem 1.3rem;
            margin: 0.6rem 0 1.4rem 0;
        }
        .figbox.status-good     { border-left-color: var(--good); }
        .figbox.status-moderate { border-left-color: var(--moderate); }
        .figbox.status-bad      { border-left-color: var(--bad); }
        .figbox .label {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: var(--muted);
            font-family: -apple-system,"Segoe UI",Roboto,sans-serif;
        }
        .figbox .stat {
            font-family: "SF Mono","Consolas","Liberation Mono",monospace;
            font-size: 1.9rem; font-weight: 600; color: var(--ink);
            line-height: 1.25; margin-top: 0.2rem;
        }
        .figbox .sub {
            color: var(--muted); font-size: 0.9rem; margin-top: 0.15rem;
            font-family: -apple-system,"Segoe UI",Roboto,sans-serif;
        }

        /* ---- landing stat cards (outcome at a glance, no prose) ---- */
        .statcard {
            border: 1px solid var(--rule);
            border-top: 3px solid var(--ink);
            padding: 1.1rem 1.2rem 1rem;
            height: 100%;
        }
        .statcard .num {
            font-family: "SF Mono","Consolas","Liberation Mono",monospace;
            font-size: 2.2rem; font-weight: 600; color: var(--ink); line-height: 1;
        }
        .statcard .cap {
            font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--muted); margin-top: 0.5rem;
            font-family: -apple-system,"Segoe UI",Roboto,sans-serif;
        }

        /* ---- sensor tile + status pill (env. monitoring page) ---- */
        .sensor-tile {
            border: 1px solid var(--rule);
            border-top: 3px solid var(--rule-2);
            padding: 0.9rem 1rem;
            height: 100%;
        }
        .sensor-tile .name {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--muted);
            font-family: -apple-system,"Segoe UI",Roboto,sans-serif;
        }
        .sensor-tile .value {
            font-family: "SF Mono","Consolas","Liberation Mono",monospace;
            font-size: 1.6rem; font-weight: 600; color: var(--ink);
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
            font-family: -apple-system,"Segoe UI",Roboto,sans-serif;
        }
        .pill-good     { background: rgba(26,127,55,0.12);  color: var(--good); }
        .pill-moderate { background: rgba(181,129,5,0.14);  color: var(--moderate); }
        .pill-bad      { background: rgba(201,42,42,0.12);  color: var(--bad); }
        .pill-unknown  { background: rgba(107,107,107,0.12); color: var(--muted); }

        /* ---- compact zone table (Swiss ruled) ---- */
        table.zones { width: 100%; border-collapse: collapse; margin-top: 0.2rem; }
        table.zones th, table.zones td {
            text-align: left; padding: 0.5rem 0.4rem;
            border-bottom: 1px solid var(--rule);
            font-family: -apple-system,"Segoe UI",Roboto,sans-serif; font-size: 0.9rem;
        }
        table.zones th {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
            color: var(--muted); border-bottom: 1px solid var(--rule-2);
        }
        table.zones td.cell {
            font-family: "SF Mono","Consolas",monospace; text-align: right;
            font-weight: 600;
        }
        table.zones td.dist { color: var(--muted); }

        /* ---- plotly charts: no card frame, just a hairline top rule ---- */
        [data-testid="stPlotlyChart"] {
            border-top: 1px solid var(--rule);
            padding-top: 0.5rem;
        }

        /* ---- sidebar: paper with a hairline divider ---- */
        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--rule-2);
        }
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
        .side-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 1.35rem; font-weight: 600; color: var(--ink);
            padding-bottom: 0.35rem; margin-bottom: 0.6rem;
            border-bottom: 2px solid var(--ink);
        }
        [data-testid="stSidebar"] h3 {
            font-size: 1rem; margin-top: 1.1rem; margin-bottom: 0.2rem;
        }

        code { font-family: "SF Mono","Consolas",monospace; background: var(--panel); }
        </style>
        """,
        unsafe_allow_html=True,
    )
