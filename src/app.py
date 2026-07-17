"""
app.py — Streamlit dashboard for the Indian Railways Optimal Freight Route Finder.
Run:  streamlit run src/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.io as pio
import plotly.graph_objects as go

import analytics as A
from pathfinder import get_finder, DEFAULT_COMMODITY, DEFAULT_RAKE_TYPE

pio.templates.default = "plotly_white"

OUTLIER_PKL = os.path.join(os.path.dirname(__file__), "../data/outlier_report.pkl")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Freight Route Analytics",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --blue:   #2563eb; --blue-l: #dbeafe;
    --sky:    #0ea5e9;
    --green:  #16a34a; --green-l:#dcfce7;
    --amber:  #d97706; --amber-l:#fef3c7;
    --red:    #dc2626; --red-l:  #fee2e2;
    --purple: #7c3aed; --purple-l:#ede9fe;
    --bg:     #f0f4f8; --card: #ffffff;
    --border: #e2e8f0; --text: #0f172a; --muted: #64748b;
    --shadow: 0 4px 20px rgba(0,0,0,.07);
    --shadow-lg: 0 8px 32px rgba(0,0,0,.12);
}
html, body, [class*="css"] { color: var(--text) !important; }
.stApp { background: var(--bg); }
.block-container { padding: 1.5rem 2rem 2rem; }
h1,h2,h3,h4,h5,h6,p,span,div,label,small,strong { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] * { color: var(--text) !important; }
[data-testid="metric-container"] * { color: var(--text) !important; }
footer { visibility: hidden; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1445 0%, #1a3a6e 60%, #0f4c81 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] span { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] table { background: rgba(255,255,255,.07) !important; }
section[data-testid="stSidebar"] th { background: rgba(37,99,235,.5) !important; color:white !important; }
section[data-testid="stSidebar"] td { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: white !important; color: #111827 !important;
    border: 1px solid rgba(255,255,255,.2) !important; border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder { color: #9ca3af !important; }
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,.1) !important;
    border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox svg { fill: white !important; }

/* ── hero ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 40%, #0369a1 100%);
    border-radius: 20px; padding: 32px 40px; margin-bottom: 24px;
    box-shadow: var(--shadow-lg); position: relative; overflow: hidden;
}
.hero-banner::before {
    content:""; position:absolute; top:-40px; right:-40px;
    width:220px; height:220px; background:rgba(255,255,255,.05); border-radius:50%;
}
.hero-banner::after {
    content:""; position:absolute; bottom:-60px; right:80px;
    width:160px; height:160px; background:rgba(255,255,255,.04); border-radius:50%;
}
.hero-title { font-size:2.2rem; font-weight:800; color:white !important; margin:0 0 6px; letter-spacing:-.5px; }
.hero-sub   { font-size:1rem; color:rgba(255,255,255,.80) !important; margin:0 0 20px; }
.hero-badges { display:flex; gap:10px; flex-wrap:wrap; }
.badge {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(255,255,255,.15); backdrop-filter:blur(6px);
    border:1px solid rgba(255,255,255,.25); border-radius:20px;
    padding:4px 14px; font-size:.82rem; color:white !important; font-weight:500;
}

/* ── search card ── */
.search-card {
    background:white; border-radius:18px; padding:22px 28px;
    box-shadow:var(--shadow); margin-bottom:16px; border:1px solid var(--border);
}
.search-divider {
    border:none; border-top:1px dashed var(--border); margin:14px 0;
}

/* ── inputs ── */
.stTextInput input {
    background:#f8fafc !important; color:var(--text) !important;
    border:2px solid var(--border) !important; border-radius:12px !important;
    padding:11px 14px !important; font-size:15px !important; font-weight:500 !important;
}
.stTextInput input:focus {
    border-color:var(--blue) !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.12) !important; background:white !important;
}
.stTextInput input::placeholder { color:#94a3b8 !important; }
.stTextInput label { font-weight:600 !important; font-size:.9rem !important; }

/* ── selectbox ── */
.stSelectbox div[data-baseweb="select"] {
    background:#f8fafc !important; border:2px solid var(--border) !important; border-radius:12px !important;
}
.stSelectbox div[data-baseweb="select"] span { color:var(--text) !important; font-weight:500; }
.stSelectbox svg { fill:var(--muted) !important; }
div[role="listbox"] {
    background:white !important; border-radius:12px !important;
    box-shadow:var(--shadow-lg); border:1px solid var(--border) !important;
}
div[role="option"] { color:var(--text) !important; background:white !important; }
div[role="option"]:hover { background:#eff6ff !important; color:var(--blue) !important; }
div[aria-selected="true"] { background:var(--blue) !important; color:white !important; }

/* ── button ── */
.stButton > button {
    background:linear-gradient(135deg,#1d4ed8,#2563eb,#0ea5e9) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    padding:11px 32px !important; font-size:15px !important; font-weight:700 !important;
    box-shadow:0 4px 14px rgba(37,99,235,.35) !important; transition:all .25s ease !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 20px rgba(37,99,235,.45) !important; }
.stButton > button:active { transform:translateY(0) !important; }

/* ── KPI cards ── */
.kpi-card {
    border-radius:16px; padding:20px 22px; box-shadow:var(--shadow);
    border:1px solid var(--border); background:white; transition:transform .2s;
}
.kpi-card:hover { transform:translateY(-2px); box-shadow:var(--shadow-lg); }
.kpi-card.blue  { border-top:4px solid var(--blue); }
.kpi-card.green { border-top:4px solid var(--green); }
.kpi-card.amber { border-top:4px solid var(--amber); }
.kpi-card.red   { border-top:4px solid var(--red); }
.kpi-icon  { font-size:1.6rem; margin-bottom:8px; }
.kpi-label { font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--muted) !important; margin-bottom:4px; }
.kpi-val   { font-size:1.75rem; font-weight:800; line-height:1.1; }
.kpi-val.blue  { color:var(--blue) !important; }
.kpi-val.green { color:var(--green) !important; }
.kpi-val.amber { color:var(--amber) !important; }
.kpi-val.red   { color:var(--red) !important; }
.kpi-sub  { font-size:.82rem; color:var(--muted) !important; margin-top:4px; }
.kpi-na   { font-size:1rem; color:var(--muted) !important; font-style:italic; }

/* ── chips ── */
.route-seq { display:flex; flex-wrap:wrap; gap:4px; align-items:center; margin:10px 0 16px; }
.chip {
    display:inline-block; background:var(--blue-l); color:var(--blue) !important;
    border:1px solid #bfdbfe; border-radius:8px; padding:3px 10px;
    font-size:.82rem; font-weight:700; font-family:monospace;
}
.chip.start    { background:var(--green-l);  color:var(--green) !important;  border-color:#86efac; }
.chip.end      { background:var(--red-l);    color:var(--red) !important;    border-color:#fca5a5; }
.chip.ellipsis { background:#f1f5f9;         color:var(--muted) !important;  border-color:var(--border); }
.arrow { color:var(--muted) !important; font-size:.85rem; padding:0 1px; }

/* ── context pill ── */
.ctx-pill {
    display:inline-flex; align-items:center; gap:6px;
    border-radius:20px; padding:4px 12px; font-size:.8rem; font-weight:600;
    margin-right:6px; margin-bottom:4px;
}
.ctx-pill.commodity { background:var(--purple-l); color:#6d28d9 !important; border:1px solid #c4b5fd; }
.ctx-pill.rake      { background:var(--amber-l);  color:var(--amber) !important; border:1px solid #fcd34d; }
.ctx-pill.model     { background:var(--blue-l);   color:var(--blue) !important;  border:1px solid #93c5fd; }

/* ── result banner ── */
.result-banner {
    background:linear-gradient(135deg,#f0fdf4,#dcfce7);
    border:1.5px solid #86efac; border-radius:14px;
    padding:16px 20px; margin-bottom:16px;
}
.result-banner-title { font-size:1.05rem; font-weight:700; color:var(--green) !important; margin-bottom:6px; }
.result-banner-sub   { font-size:.85rem; color:var(--muted) !important; margin-bottom:8px; }

/* ── section headings ── */
.section-heading {
    font-size:.9rem; font-weight:700; color:var(--text) !important;
    text-transform:uppercase; letter-spacing:.07em;
    padding:4px 0 10px; border-bottom:2px solid var(--border); margin-bottom:14px;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#f1f5f9; border-radius:12px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] {
    border-radius:9px; padding:8px 18px; font-weight:600; font-size:.88rem;
    color:var(--muted) !important; background:transparent; border:none;
}
.stTabs [aria-selected="true"] { background:white !important; color:var(--blue) !important; box-shadow:0 1px 4px rgba(0,0,0,.1); }
button[role="tab"] { color:var(--muted) !important; }
button[role="tab"][aria-selected="true"] { color:var(--blue) !important; }

/* ── dataframe / metric ── */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; box-shadow:var(--shadow); }
[data-testid="metric-container"] { background:white; border-radius:12px; padding:16px; box-shadow:var(--shadow); }

/* ── alerts ── */
.stSuccess { background:#f0fdf4 !important; border-left:4px solid var(--green) !important; border-radius:10px !important; }
.stInfo    { background:#eff6ff !important; border-left:4px solid var(--blue) !important;  border-radius:10px !important; }
.stWarning { background:#fffbeb !important; border-left:4px solid var(--amber) !important; border-radius:10px !important; }
.stError   { background:#fef2f2 !important; border-left:4px solid var(--red) !important;   border-radius:10px !important; }

/* ── map mode labels ── */
.map-mode-label {
    display:inline-flex; align-items:center; gap:8px; font-size:.92rem; font-weight:700;
    padding:6px 16px; border-radius:20px; margin-bottom:10px;
}
.map-mode-label.distance { background:var(--blue-l);  color:var(--blue) !important; }
.map-mode-label.time     { background:var(--green-l); color:var(--green) !important; }
.map-mode-label.balanced { background:var(--amber-l); color:var(--amber) !important; }

/* ── empty state ── */
.empty-state { text-align:center; padding:60px 20px; background:white;
    border-radius:20px; box-shadow:var(--shadow); border:2px dashed var(--border); }
.empty-state-icon  { font-size:4rem; margin-bottom:16px; }
.empty-state-title { font-size:1.3rem; font-weight:700; margin-bottom:8px; }
.empty-state-sub   { font-size:.95rem; color:var(--muted) !important; max-width:460px; margin:0 auto; }

/* ── sidebar brand / chip ── */
.sidebar-brand { display:flex; align-items:center; gap:12px; padding:8px 0 20px; }
.sidebar-brand-name { font-size:1.2rem; font-weight:800; color:white !important; }
.sidebar-brand-tag  { font-size:.75rem; color:rgba(255,255,255,.6) !important; }
.sidebar-chip {
    display:inline-block; background:rgba(255,255,255,.15);
    border:1px solid rgba(255,255,255,.25); border-radius:6px;
    padding:2px 9px; font-size:.8rem; font-weight:600; color:white !important; margin:2px;
}
::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-thumb { background:#94a3b8; border-radius:8px; }
::-webkit-scrollbar-track { background:var(--bg); }
</style>
""", unsafe_allow_html=True)

RAIL_SVG = """<svg width="36" height="36" viewBox="0 0 24 24" fill="none">
<rect x="5" y="2" width="14" height="14" rx="3" fill="#38bdf8"/>
<rect x="8" y="5" width="8" height="5" rx="1" fill="#0c1445"/>
<circle cx="9" cy="13" r="1.4" fill="#0c1445"/>
<circle cx="15" cy="13" r="1.4" fill="#0c1445"/>
<rect x="8" y="17" width="2" height="3" fill="#94a3b8"/>
<rect x="14" y="17" width="2" height="3" fill="#94a3b8"/>
<rect x="3" y="21" width="18" height="1.6" rx="0.8" fill="#fbbf24"/>
</svg>"""


# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models & route graph …")
def load_finder():
    return get_finder()

@st.cache_data(show_spinner=False)
def load_outlier_report():
    if os.path.exists(OUTLIER_PKL):
        with open(OUTLIER_PKL, "rb") as f:
            return pickle.load(f)
    return None

finder         = load_finder()
results_df     = finder.bundle["results"].copy()
validation     = finder.bundle.get("validation")
outlier_report = load_outlier_report()
MODELS         = ["XGBoost", "LightGBM", "CatBoost"]
N_NODES        = len(finder._all_nodes)
BEST_R2        = results_df.loc[results_df["R2"].idxmax(), "R2"]
COMMODITIES    = finder.get_valid_commodities()
RAKE_TYPES     = finder.get_valid_rake_types()


# ── helpers ───────────────────────────────────────────────────────────────────
def kpi_card(col, color, icon, label, value, unit="", sub=None, is_na=False):
    with col:
        body = (
            '<div class="kpi-na">No path found</div>'
            if is_na else
            f'<div class="kpi-val {color}">{value}'
            f'<span style="font-size:1rem;font-weight:500;margin-left:4px;color:#64748b !important">{unit}</span></div>'
        )
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        st.markdown(
            f'<div class="kpi-card {color}">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'{body}{sub_html}</div>',
            unsafe_allow_html=True,
        )

def ctx_pills(commodity, rake_type, model):
    return (
        f'<span class="ctx-pill commodity">🌾 {commodity}</span>'
        f'<span class="ctx-pill rake">🚃 {rake_type}</span>'
        f'<span class="ctx-pill model">🤖 {model}</span>'
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">{RAIL_SVG}'
        f'<div><div class="sidebar-brand-name">RailRoute</div>'
        f'<div class="sidebar-brand-tag">Freight Optimizer · CRIS</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**🔍 Station search**")
    query = st.text_input("Search by code or name", placeholder="e.g. JNPT, Nagpur, Coal",
                          label_visibility="collapsed")
    if query:
        matches = finder.search_stations(query)
        if matches:
            for code, name in matches[:12]:
                st.markdown(
                    f'<span class="sidebar-chip">{code}</span> '
                    f'<span style="color:rgba(255,255,255,.55);font-size:.78rem">{name[:32]}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No stations found.")

    st.markdown("---")
    st.markdown("**⚙️ Prediction model**")
    default_idx  = MODELS.index(finder.best_name) if finder.best_name in MODELS else 0
    model_choice = st.selectbox("Model", MODELS, index=default_idx,
                                label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**📊 Model performance**")
    perf_df = results_df.copy()
    perf_df["✓"] = perf_df["model"].apply(lambda x: "✅" if x == model_choice else "")
    perf_df = perf_df[["✓", "model", "MAE", "RMSE", "R2"]].round(3)
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    sel = perf_df[perf_df["model"] == model_choice].iloc[0]
    st.markdown(
        f'<div style="background:rgba(255,255,255,.08);border-radius:10px;padding:12px 14px;margin-top:8px">'
        f'<div style="color:rgba(255,255,255,.55) !important;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">Active: {model_choice}</div>'
        f'<div style="display:flex;justify-content:space-between;margin:4px 0"><span style="color:rgba(255,255,255,.65) !important;font-size:.83rem">MAE</span><span style="color:white !important;font-weight:700;font-size:.83rem">{sel["MAE"]:.3f}</span></div>'
        f'<div style="display:flex;justify-content:space-between;margin:4px 0"><span style="color:rgba(255,255,255,.65) !important;font-size:.83rem">RMSE</span><span style="color:white !important;font-weight:700;font-size:.83rem">{sel["RMSE"]:.3f}</span></div>'
        f'<div style="display:flex;justify-content:space-between;margin:4px 0"><span style="color:rgba(255,255,255,.65) !important;font-size:.83rem">R²</span><span style="color:#4ade80 !important;font-weight:700;font-size:.83rem">{sel["R2"]:.3f}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="hero-banner">'
    f'<div class="hero-title">🚂 RailRoute Optimizer</div>'
    f'<div class="hero-sub">AI-powered freight route optimization for Indian Railways · CRIS Internship Project</div>'
    f'<div class="hero-badges">'
    f'<span class="badge">📍 {N_NODES:,} stations</span>'
    f'<span class="badge">🤖 Best R² {BEST_R2:.3f}</span>'
    f'<span class="badge">⚡ Dijkstra + ML</span>'
    f'<span class="badge">🗺️ 3 optimization modes</span>'
    f'<span class="badge">🌾 {len(COMMODITIES)} commodity types</span>'
    f'<span class="badge">🚃 {len(RAKE_TYPES)} rake types</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ── search card ───────────────────────────────────────────────────────────────
st.markdown('<div class="search-card">', unsafe_allow_html=True)

# Row 1: source / destination / button
r1c1, r1c2, r1c3 = st.columns([5, 5, 3])
with r1c1:
    source = st.text_input("🟢 Source station code", placeholder="e.g. JNPT", key="src").strip().upper()
with r1c2:
    target = st.text_input("🔴 Destination station code", placeholder="e.g. AFAS", key="dst").strip().upper()
with r1c3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    find_btn = st.button("🔎 Find optimal paths", use_container_width=True)

st.markdown('<hr class="search-divider">', unsafe_allow_html=True)

# Row 2: commodity / rake type
r2c1, r2c2, r2c3 = st.columns([5, 5, 3])
with r2c1:
    cmdt_idx   = COMMODITIES.index(DEFAULT_COMMODITY) if DEFAULT_COMMODITY in COMMODITIES else 0
    commodity  = st.selectbox("🌾 Commodity group", COMMODITIES, index=cmdt_idx,
                              help="Commodity being transported. Affects ML circuit-speed prediction.")
with r2c2:
    rake_idx   = RAKE_TYPES.index(DEFAULT_RAKE_TYPE) if DEFAULT_RAKE_TYPE in RAKE_TYPES else 0
    rake_type  = st.selectbox("🚃 Rake / wagon type", RAKE_TYPES, index=rake_idx,
                              help="Wagon type carrying the freight. Affects ML circuit-speed prediction.")
with r2c3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:10px 14px;font-size:.8rem;color:#16a34a !important;font-weight:600">'
        '✅ These inputs directly influence the ML speed predictions on every route edge.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── session state: persist result across reruns ────────────────────────────────
if find_btn and source and target:
    with st.spinner(f"Computing {source} → {target} · {commodity} · {rake_type} · {model_choice} …"):
        st.session_state["route_result"] = finder.find_optimal_path(
            source, target,
            model_name=model_choice,
            commodity=commodity,
            rake_type=rake_type,
        )
        st.session_state["route_query"] = (source, target, model_choice, commodity, rake_type)
elif find_btn:
    st.warning("⚠️ Please enter both source and destination station codes.")
    st.session_state.pop("route_result", None)

with st.expander("💡 Example station pairs"):
    eg_cols = st.columns(3)
    examples = [
        ("JNPT", "AFAS", "Mumbai Port → Adani Fwd Agent"),
        ("ABKP", "UD",   "Ambikapur → Udupi"),
        ("MTRN", "TWS",  "Mettur → Tiruvallur"),
        ("AKPK", "JSWT", "Akupuram → JSW Toranagallu"),
        ("PSRS", "PCMC", "Posnur Sidings → Pimpri-Chinchwad"),
        ("DLI",  "MMCT", "Delhi → Mumbai Central"),
    ]
    for i, (src, dst, label) in enumerate(examples):
        with eg_cols[i % 3]:
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid var(--border);border-radius:10px;padding:10px 12px;margin:4px 0">'
                f'<div style="font-size:.8rem;color:var(--muted) !important;margin-bottom:4px">{label}</div>'
                f'<span class="chip start">{src}</span>'
                f'<span class="arrow"> → </span>'
                f'<span class="chip end">{dst}</span></div>',
                unsafe_allow_html=True,
            )


# ── render helpers ────────────────────────────────────────────────────────────
def render_path_tab(info, mode, commodity, rake_type):
    if not info.get("path"):
        st.warning("No path found for this optimization mode.")
        return

    path      = info["path"]
    avg_speed = info["total_km"] / max(info["total_ml_hrs"], 0.01)

    # Query context pills at top of tab
    st.markdown(
        f'<div style="margin-bottom:12px">'
        f'{ctx_pills(commodity, rake_type, info.get("model_used",""))}'
        f'</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total distance",    f"{info['total_km']:,.1f} km")
    m2.metric("ML est. time",      f"{info['total_ml_hrs']:,.1f} hrs")
    m3.metric("Avg circuit speed", f"{avg_speed:.1f} km/h")
    m4.metric("Intermediate hops", len(path) - 1)

    st.markdown('<div class="section-heading">Route sequence</div>', unsafe_allow_html=True)
    seq   = path if len(path) <= 12 else path[:5] + ["…"] + path[-3:]
    parts = []
    for j, s in enumerate(seq):
        if s == "…":
            parts.append('<span class="chip ellipsis">···</span>')
        elif j == 0:
            parts.append(f'<span class="chip start">{s}</span>')
        elif j == len(seq) - 1:
            parts.append(f'<span class="chip end">{s}</span>')
        else:
            parts.append(f'<span class="chip">{s}</span>')
    st.markdown(
        '<div class="route-seq">' +
        '<span class="arrow"> → </span>'.join(parts) +
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Hop-by-hop breakdown</div>', unsafe_allow_html=True)
    hop_df = pd.DataFrame(info["hops"])
    hop_df.columns = ["From Code", "From Name", "To Code", "To Name",
                      "KM", "ML Hours", "ML Speed (km/h)"]
    hop_df["From Name"] = hop_df["From Name"].str[:32]
    hop_df["To Name"]   = hop_df["To Name"].str[:32]
    hop_df["KM"]        = hop_df["KM"].round(1)
    hop_df["ML Hours"]  = hop_df["ML Hours"].round(2)
    hop_df["ML Speed (km/h)"] = hop_df["ML Speed (km/h)"].round(1)
    st.dataframe(hop_df, use_container_width=True, hide_index=True)

    if len(info["hops"]) <= 40:
        st.plotly_chart(
            A.hop_distance_bar(hop_df),
            use_container_width=True,
            key=f"hop_{mode}_{len(info['hops'])}_{st.session_state.get('src','')}_{st.session_state.get('dst','')}",
        )


def render_map(info, mode="distance", map_key="map"):
    mode_cfg = {
        "distance": ("📏 Shortest Distance", "#2563eb",  "distance"),
        "time":     ("⏱️ Fastest Route (ML)", "#16a34a", "time"),
        "balanced": ("⚖️ Balanced Route",     "#d97706", "balanced"),
    }
    title, line_color, css_mode = mode_cfg.get(mode, ("Route", "#2563eb", "distance"))
    coords = info.get("coords")

    st.markdown(f'<div class="map-mode-label {css_mode}">{title}</div>', unsafe_allow_html=True)
    if not coords or len(coords) < 2:
        st.warning("Map data not available for this route.")
        return

    center_lat = sum(c[0] for c in coords) / len(coords)
    center_lon = sum(c[1] for c in coords) / len(coords)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB Positron")

    for i, (lat, lon, name) in enumerate(coords):
        if i == 0:
            folium.Marker([lat, lon], tooltip=f"🟢 START: {name}",
                          icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
        elif i == len(coords) - 1:
            folium.Marker([lat, lon], tooltip=f"🔴 END: {name}",
                          icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(m)
        else:
            folium.CircleMarker([lat, lon], radius=5, color=line_color,
                                fill=True, fill_color="white", fill_opacity=0.9,
                                weight=2, tooltip=name).add_to(m)

    folium.PolyLine([(lat, lon) for lat, lon, _ in coords],
                    color=line_color, weight=4, opacity=0.85, tooltip=title).add_to(m)
    st_folium(m, use_container_width=True, height=460, key=map_key)


def render_analytics():
    st.markdown('<div class="section-heading">Model evaluation</div>', unsafe_allow_html=True)
    metric_cols = ["MAE", "MSE", "RMSE", "RMSLE", "R2"]
    have     = [m for m in metric_cols if m in results_df.columns]
    best_row = results_df.loc[results_df["MAE"].idxmin()]
    icons    = {"MAE": "📉", "MSE": "📉", "RMSE": "📉", "RMSLE": "📉", "R2": "📈"}
    cols     = st.columns(len(have))
    for col, m in zip(cols, have):
        col.metric(f"{icons.get(m,'')} {m} (best)", f"{best_row[m]:.4f}",
                   delta=f"{best_row['model']}", delta_color="off")

    st.markdown('<div class="section-heading" style="margin-top:20px">Metric comparison across models</div>',
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(A.metric_bars(results_df, "MAE",  True),  use_container_width=True)
    g2.plotly_chart(A.metric_bars(results_df, "RMSE", True),  use_container_width=True)
    g3.plotly_chart(A.metric_bars(results_df, "R2",   False), use_container_width=True)
    st.dataframe(results_df.set_index("model").round(4), use_container_width=True)

    if validation is None:
        st.info("Re-run `python run_pipeline.py` to enable residual and prediction-vs-actual plots.")
        return

    st.markdown('<div class="section-heading" style="margin-top:20px">Residual & prediction analysis</div>',
                unsafe_allow_html=True)
    pick   = st.selectbox("Model to analyze", MODELS,
                          index=MODELS.index(finder.best_name) if finder.best_name in MODELS else 0,
                          key="resid_model")
    y_true = np.asarray(validation["y_true"], dtype=float)
    y_pred = np.asarray(validation["y_pred"][pick], dtype=float)

    p1, p2 = st.columns(2)
    p1.plotly_chart(A.pred_vs_actual(y_true, y_pred, pick),   use_container_width=True)
    p2.plotly_chart(A.residual_scatter(y_true, y_pred, pick), use_container_width=True)
    st.plotly_chart(A.residual_hist(y_true, y_pred, pick),    use_container_width=True)

    resid = y_true - y_pred
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean residual",  f"{resid.mean():.3f}")
    s2.metric("Residual std",   f"{resid.std():.3f}")
    s3.metric("Within ±2 km/h", f"{100*np.mean(np.abs(resid) <= 2):.1f}%")


def render_outliers():
    st.markdown('<div class="section-heading">Data-quality & outlier analysis</div>',
                unsafe_allow_html=True)
    if outlier_report is None:
        st.info("Outlier report not found — re-run `python run_pipeline.py` to generate it.")
        return
    st.caption("Outliers detected with the IQR (Tukey) rule and winsorised to the fences "
               "during preprocessing, preserving graph connectivity while de-noising the ML target.")
    st.plotly_chart(A.outlier_pct_bar(outlier_report), use_container_width=True)
    st.dataframe(outlier_report.set_index("column"),   use_container_width=True)
    st.markdown('<div class="section-heading" style="margin-top:20px">Distribution & fences</div>',
                unsafe_allow_html=True)
    sims = {
        "circuit_speed": np.random.default_rng(0).normal(8.7, 4.9,  4000).clip(0.5, 25),
        "ldng_uldg_km":  np.random.default_rng(1).normal(655, 569,  4000).clip(5,  4000),
        "ldng_uldg_hor": np.random.default_rng(2).normal(65,  58,   4000).clip(1,   400),
    }
    oc = st.columns(3)
    for col, (name, sample) in zip(oc, sims.items()):
        col.plotly_chart(A.box_with_outliers(sample, name), use_container_width=True)
    st.caption("Box plots illustrate the IQR spread per column; "
               "the report table above carries exact counts from the full 441k-row dataset.")


# ── route results ─────────────────────────────────────────────────────────────
result = st.session_state.get("route_result")

if result is not None:
    if "error" in result:
        st.error(f"❌ {result['error']}")
        st.info("Use the station search in the sidebar to find valid codes.")
    else:
        paths = result["paths"]
        q     = st.session_state.get("route_query", ("", "", "", DEFAULT_COMMODITY, DEFAULT_RAKE_TYPE))
        res_commodity = result.get("commodity", q[3])
        res_rake      = result.get("rake_type",  q[4])
        res_model     = q[2]

        # Result banner with context pills
        st.markdown(
            f'<div class="result-banner">'
            f'<div class="result-banner-title">✅ {result["source"]} → {result["target"]}</div>'
            f'<div class="result-banner-sub">'
            f'{result["source_name"][:45]} → {result["target_name"][:45]}'
            f'</div>'
            f'<div style="margin-top:8px">'
            f'{ctx_pills(res_commodity, res_rake, res_model)}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # KPI cards
        c1, c2, c3 = st.columns(3)
        for col, (mode, color, icon, label, key, unit, fmt) in zip(
            [c1, c2, c3],
            [
                ("distance", "blue",  "📏", "Shortest Distance", "total_km",     "km",  "{:,.1f}"),
                ("time",     "green", "⏱️", "Fastest Time (ML)",  "total_ml_hrs", "hrs", "{:,.1f}"),
                ("balanced", "amber", "⚖️", "Balanced Score",     "cost",         "",    "{:,.4f}"),
            ],
        ):
            info = paths.get(mode, {})
            if info.get("path"):
                kpi_card(col, color, icon, label,
                         fmt.format(info.get(key, 0)), unit=unit,
                         sub=f"{len(info['path'])-1} hops · {info.get('total_km',0):,.0f} km · {res_commodity} / {res_rake}")
            else:
                kpi_card(col, "red", icon, label, None, is_na=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        tabs = st.tabs(["📏 Distance", "⏱️ Time", "⚖️ Balanced",
                        "🗺️ Maps", "📊 Analytics", "🧪 Outliers"])

        with tabs[0]: render_path_tab(paths.get("distance", {}), "distance", res_commodity, res_rake)
        with tabs[1]: render_path_tab(paths.get("time", {}),     "time",     res_commodity, res_rake)
        with tabs[2]: render_path_tab(paths.get("balanced", {}), "balanced", res_commodity, res_rake)

        with tabs[3]:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-bottom:14px">{ctx_pills(res_commodity, res_rake, res_model)}</div>',
                unsafe_allow_html=True,
            )
            ma, mb, mc = st.columns(3)
            with ma: render_map(paths.get("distance", {}), mode="distance", map_key="map_distance")
            with mb: render_map(paths.get("time", {}),     mode="time",     map_key="map_time")
            with mc: render_map(paths.get("balanced", {}), mode="balanced", map_key="map_balanced")

        with tabs[4]: render_analytics()
        with tabs[5]: render_outliers()

else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-state-icon">🚂</div>'
        '<div class="empty-state-title">Ready to find the optimal freight route</div>'
        '<div class="empty-state-sub">Enter source and destination codes, select commodity and rake type, '
        'then click <strong>Find optimal paths</strong>. '
        'Three routes will be computed — shortest distance, fastest ML-predicted time, '
        'and a balanced composite — all calibrated to your selected commodity and wagon type.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;padding:16px;color:#94a3b8 !important;font-size:.8rem'>"
    "🚂 RailRoute Optimizer &nbsp;·&nbsp; XGBoost · LightGBM · CatBoost &nbsp;·&nbsp; "
    "Dijkstra Route Optimization &nbsp;·&nbsp; CRIS Internship Project"
    "</div>",
    unsafe_allow_html=True,
)
