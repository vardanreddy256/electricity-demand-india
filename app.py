"""
India Power Grid Analytics — Multi-Granularity Demand Forecasting
=================================================================
Premium Streamlit dashboard with:
  • Historical trend analysis (1W / 1M / 1Y / All)
  • Seasonality patterns
  • Model performance scorecard
  • Multi-Horizon Forecasting: Second → Minute → Hour → Day → Week → Month → Year
"""

import os
import sys
import warnings
import base64
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "India Power Grid Analytics",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
    menu_items  = {
        "Get help":    "https://github.com",
        "Report a bug": None,
        "About": "India Power Grid Analytics — AI-Driven Electricity Demand Forecasting"
    }
)

# ══════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "enhanced_hourly_electricity_dataset.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "model_dict.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")
ENCODER_PATH  = os.path.join(BASE_DIR, "season_encoder.pkl")
META_PATH     = os.path.join(BASE_DIR, "model_meta.pkl")
BANNER_PATH   = os.path.join(BASE_DIR, "assets", "background.jpg")
SRC_DIR       = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme
# ══════════════════════════════════════════════════════════════════════
def inject_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(135deg, #020b18 0%, #061221 40%, #09192e 100%);
        min-height: 100vh;
    }

    /* ── Header ──────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(90deg, rgba(0,180,255,0.08) 0%, rgba(100,60,255,0.08) 100%);
        border: 1px solid rgba(0,180,255,0.15);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c6ff, #9b59ff, #ff4fa0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .main-subtitle {
        color: #8899bb;
        font-size: 0.95rem;
        margin: 6px 0 0 2px;
        font-weight: 400;
    }

    /* ── Metric Cards ─────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,120,200,0.12) 0%, rgba(80,40,180,0.12) 100%);
        border: 1px solid rgba(0,180,255,0.2);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0,180,255,0.45);
    }
    .metric-label {
        color: #7a92b8;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #00d4ff;
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .metric-unit {
        color: #4a90d9;
        font-size: 0.7rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .metric-delta {
        color: #00e676;
        font-size: 0.75rem;
        margin-top: 6px;
    }

    /* ── Section Headers ──────────────────────────────────── */
    .section-header {
        color: #c0d4ff;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0,180,255,0.15);
    }

    /* ── Granularity Badge ────────────────────────────────── */
    .gran-badge {
        display: inline-block;
        background: linear-gradient(90deg, #0066cc33, #6600cc33);
        border: 1px solid rgba(0,150,255,0.35);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #66ccff;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .synth-badge {
        display: inline-block;
        background: rgba(255,160,0,0.12);
        border: 1px solid rgba(255,160,0,0.35);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        color: #ffaa33;
        margin-left: 8px;
    }

    /* ── Streamlit overrides ──────────────────────────────── */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00d4ff !important;
        font-size: 1.5rem !important;
    }
    div[data-testid="stMetricLabel"] { color: #7a92b8 !important; }
    div[data-testid="stMetricDelta"] { color: #00e676 !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030e1f 0%, #050f20 100%) !important;
        border-right: 1px solid rgba(0,100,200,0.2);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label { color: #aabbcc !important; }

    .stTabs [data-baseweb="tab-list"] { background: rgba(5,15,35,0.8) !important; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { color: #7a92b8 !important; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #00c6ff !important; background: rgba(0,150,255,0.1) !important; border-radius: 8px; }

    .stButton > button {
        background: linear-gradient(90deg, #0055cc, #6600cc) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 28px !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0077ff, #8800ff) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,100,255,0.35) !important;
    }

    .stDataFrame { border: 1px solid rgba(0,100,200,0.2) !important; border-radius: 10px !important; }

    .info-box {
        background: rgba(0,120,255,0.08);
        border-left: 3px solid #0080ff;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #8aaddd;
        margin: 8px 0;
    }

    div[data-testid="stAlert"] { border-radius: 10px !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_css()


# ══════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
    return df


@st.cache_resource(show_spinner=False)
def load_resources():
    models   = joblib.load(MODEL_PATH)   if os.path.exists(MODEL_PATH)    else None
    features = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None
    le       = joblib.load(ENCODER_PATH)  if os.path.exists(ENCODER_PATH)  else None
    meta     = joblib.load(META_PATH)     if os.path.exists(META_PATH)      else {}
    return models, features, le, meta


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <p class="main-title">⚡ India Power Grid Analytics</p>
  <p class="main-subtitle">AI-Driven Multi-Granularity Electricity Demand Forecasting — XGBoost Ensemble</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════
with st.spinner("Loading models and data..."):
    df                           = load_data()
    models, feature_cols, le, meta = load_resources()

if df is None or models is None:
    st.error(f"⚠️ Model or Data file not found. Run `python src/train_multi_granularity.py` first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Region selector
    available_regions = sorted(models.keys(), key=lambda x: (0 if "National" in x else 1, x))
    selected_region   = st.selectbox("🗺️ Region / Grid", available_regions, key="region_sel")

    st.markdown("---")
    st.markdown("### 🕐 Forecast Settings")

    GRANULARITIES = ["Second", "Minute", "Hour", "Day", "Week", "Month", "Year"]
    gran_icons    = {"Second": "⏱️", "Minute": "🕐", "Hour": "⏰",
                     "Day": "📅", "Week": "📆", "Month": "🗓️", "Year": "📊"}

    selected_gran = st.select_slider(
        "Granularity",
        options=GRANULARITIES,
        value="Hour",
        key="gran_slider"
    )

    forecast_date = st.date_input("Start Date", value=pd.Timestamp.now().date(), key="fdate")
    forecast_time = st.time_input("Start Time", value=pd.Timestamp.now().replace(minute=0, second=0).time(), key="ftime")

    gran_defaults = {"Second": 60, "Minute": 60, "Hour": 24,
                     "Day": 7, "Week": 4, "Month": 12, "Year": 12}
    gran_max      = {"Second": 120, "Minute": 120, "Hour": 72,
                     "Day": 30, "Week": 12, "Month": 24, "Year": 12}

    n_periods = st.slider(
        "Periods to Forecast",
        min_value = 1,
        max_value = gran_max[selected_gran],
        value     = gran_defaults[selected_gran],
        key       = "n_periods_slider",
        help      = f"Number of {selected_gran.lower()} intervals"
    )

    run_forecast = st.button(f"🔮 Run {gran_icons.get(selected_gran,'')} {selected_gran} Forecast",
                              type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='color:#5a7090; font-size:0.78rem; line-height:1.6'>
    <b style="color:#6090c0">Model:</b> XGBoost + Random Forest Ensemble<br>
    <b style="color:#6090c0">Data:</b> India Hourly Grid Data<br>
    <b style="color:#6090c0">Features:</b> 35+ temporal, cyclical & weather<br>
    <b style="color:#6090c0">Lag:</b> Up to 168h (1 week look-back)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# LOAD FORECASTER
# ══════════════════════════════════════════════════════════════════════
try:
    from multi_granularity_forecaster import GranularityForecaster
    forecaster = GranularityForecaster(models, feature_cols, le, meta, df)
except ImportError as e:
    st.error(f"Could not load forecasting engine: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# KPI BAR — Selected Region
# ══════════════════════════════════════════════════════════════════════
region_series = df[selected_region].dropna() if selected_region in df.columns else None

if region_series is not None:
    latest  = region_series.iloc[-1]
    avg_val = region_series.mean()
    peak    = region_series.max()
    trough  = region_series.min()
    n_days  = (region_series.index[-1] - region_series.index[0]).days

    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (col1, "Latest Demand",  f"{latest:,.0f}", "MW", "● Live"),
        (col2, "Average Load",   f"{avg_val:,.0f}", "MW", f"{n_days} days"),
        (col3, "Peak Demand",    f"{peak:,.0f}",    "MW", f"↑ All-time high"),
        (col4, "Min Demand",     f"{trough:,.0f}",  "MW", "↓ Night trough"),
        (col5, "Data Points",    f"{len(region_series):,}", "hours", "Hourly resolution"),
    ]
    for col, label, val, unit, delta in kpis:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
          <div class="metric-unit">{unit}</div>
          <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historical Trends",
    "🍂 Seasonality",
    "🕐 Multi-Horizon Forecast",
    "🏆 Model Performance"
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — Historical Trends
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Time-Series Demand Analysis</p>', unsafe_allow_html=True)

    if region_series is not None:
        c1, c2 = st.columns([3, 1])
        with c1:
            range_opt = st.radio("Time Range", ["1 Week", "1 Month", "3 Months", "1 Year", "All"],
                                  horizontal=True, key="hist_range")
        with c2:
            show_ma = st.checkbox("Show 7-day MA", value=True, key="show_ma")

        rng_map = {"1 Week": 24*7, "1 Month": 24*30, "3 Months": 24*90, "1 Year": 24*365}
        plot_s  = region_series.tail(rng_map[range_opt]) if range_opt != "All" else region_series

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_s.index, y=plot_s.values,
            mode="lines", name="Demand",
            line=dict(color="#00c6ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,180,255,0.05)"
        ))
        if show_ma and len(plot_s) > 168:
            ma = plot_s.rolling(168).mean()
            fig.add_trace(go.Scatter(
                x=ma.index, y=ma.values,
                mode="lines", name="7-day MA",
                line=dict(color="#ff9944", width=2, dash="dot")
            ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(5,15,40,0.6)",
            title=dict(text=f"<b>{selected_region}</b> — Hourly Demand",
                       font=dict(size=15, color="#aaccff")),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       title="Demand (MW)", titlefont=dict(color="#7a92b8")),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Regional comparison heatmap
        st.markdown('<p class="section-header">Regional Demand Heatmap (Last 30 Days)</p>',
                    unsafe_allow_html=True)
        demand_cols = [c for c in df.columns if "Demand" in c or "Hourly" in c]
        hmap_data   = df[demand_cols].tail(24*30)

        if len(hmap_data) > 0:
            daily = hmap_data.resample("D").mean()
            fig_h = go.Figure(go.Heatmap(
                z=daily.values.T,
                x=daily.index,
                y=[c.replace(" Hourly Demand", "").replace(" Demand", "") for c in demand_cols],
                colorscale="Viridis",
                hoverongaps=False,
                colorbar=dict(title="MW", titlefont=dict(color="#aaddff"), tickfont=dict(color="#aaddff"))
            ))
            fig_h.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)",
                title=dict(text="Daily Mean Demand by Region",
                           font=dict(size=14, color="#aaccff")),
                margin=dict(l=0, r=0, t=50, b=0),
                yaxis=dict(tickfont=dict(color="#8aaddd"))
            )
            st.plotly_chart(fig_h, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — Seasonality
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Seasonal & Temporal Patterns</p>', unsafe_allow_html=True)

    if "season" in df.columns and region_series is not None:
        cA, cB = st.columns(2)
        with cA:
            # Box plot by season
            fig_s = px.box(
                df.reset_index(), x="season", y=selected_region,
                color="season",
                color_discrete_sequence=["#00c6ff", "#ff6b6b", "#ffd93d", "#6bcb77"],
                title="Demand Distribution by Season",
                labels={selected_region: "Demand (MW)"}
            )
            fig_s.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)", showlegend=False,
                title_font=dict(color="#aaccff"),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            st.plotly_chart(fig_s, use_container_width=True)

        with cB:
            # Monthly average
            df_m  = df[selected_region].resample("ME").mean().reset_index()
            df_m.columns = ["month", "avg_demand"]
            df_m["month_name"] = df_m["month"].dt.strftime("%b %Y")
            fig_m = px.bar(
                df_m, x="month_name", y="avg_demand",
                color="avg_demand", color_continuous_scale="Blues",
                title="Monthly Average Demand",
                labels={"avg_demand": "Avg Demand (MW)", "month_name": "Month"}
            )
            fig_m.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)",
                title_font=dict(color="#aaccff"),
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # Hour-of-day profile
        df_h = df.copy()
        df_h["hour"] = df_h.index.hour
        hourly_profile = df_h.groupby(["hour", "season"])[selected_region].mean().reset_index()
        fig_hp = px.line(
            hourly_profile, x="hour", y=selected_region, color="season",
            color_discrete_sequence=["#00c6ff", "#ff6b6b", "#ffd93d", "#6bcb77"],
            title="Average Hourly Load Profile by Season",
            markers=True,
            labels={selected_region: "Avg Demand (MW)", "hour": "Hour of Day"}
        )
        fig_hp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(5,15,40,0.6)",
            title_font=dict(color="#aaccff"),
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hp, use_container_width=True)
    else:
        st.info("Season column not available in dataset.")


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — Multi-Horizon Forecast
# ─────────────────────────────────────────────────────────────────────
with tab3:
    is_synthetic = selected_gran in ["Second", "Minute"]
    is_aggregated = selected_gran in ["Day", "Week", "Month", "Year"]

    # Granularity header
    badge_html = f'<span class="gran-badge">{gran_icons.get(selected_gran,"")} {selected_gran}-Level Forecast</span>'
    if is_synthetic:
        badge_html += '<span class="synth-badge">⚠️ Synthesized from hourly model</span>'
    elif is_aggregated:
        badge_html += '<span class="synth-badge" style="background:rgba(0,200,100,0.12);border-color:rgba(0,200,100,0.35);color:#44dd88">↑ Aggregated from hourly model</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    if run_forecast:
        start_dt = pd.Timestamp(f"{forecast_date} {forecast_time}")

        with st.spinner(f"Running {selected_gran}-level forecast..."):
            try:
                forecast_df = forecaster.forecast(selected_gran, start_dt, selected_region, n_periods)
                forecast_df = GranularityForecaster.add_confidence_bands(forecast_df)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

        # ── Forecast KPIs ─────────────────────────────────────────────
        fcast_vals = forecast_df["demand_MW"]
        k1, k2, k3, k4 = st.columns(4)
        kf_items = [
            (k1, "Forecast Peak",    f"{fcast_vals.max():,.0f}",  "MW"),
            (k2, "Forecast Average", f"{fcast_vals.mean():,.0f}", "MW"),
            (k3, "Forecast Low",     f"{fcast_vals.min():,.0f}",  "MW"),
            (k4, "Periods",          f"{len(fcast_vals)}",        selected_gran + "s"),
        ]
        for col, label, val, unit in kf_items:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Forecast Chart ─────────────────────────────────────────────
        fig_f = go.Figure()

        # Confidence band
        fig_f.add_trace(go.Scatter(
            x=list(forecast_df.index) + list(forecast_df.index[::-1]),
            y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(0,150,255,0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±5% Confidence Band",
            hoverinfo="skip"
        ))
        # Upper/Lower lines
        fig_f.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["upper"],
            mode="lines", line=dict(color="rgba(0,150,255,0.2)", width=1, dash="dot"),
            name="Upper Band", showlegend=False
        ))
        fig_f.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["lower"],
            mode="lines", line=dict(color="rgba(0,150,255,0.2)", width=1, dash="dot"),
            name="Lower Band", showlegend=False
        ))
        # Forecast line
        line_color = "#ff9944" if is_synthetic else ("#00e676" if is_aggregated else "#00c6ff")
        fig_f.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["demand_MW"],
            mode="lines+markers",
            name=f"Forecast ({selected_gran})",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=5 if len(forecast_df) < 100 else 0, color=line_color),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Demand: <b>%{y:,.0f} MW</b><extra></extra>"
            )
        ))

        xlabel = "Time"
        if selected_gran == "Day":   xlabel = "Date"
        if selected_gran == "Week":  xlabel = "Week Starting"
        if selected_gran == "Month": xlabel = "Month"
        if selected_gran == "Year":  xlabel = "Month"

        fig_f.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(5,15,40,0.6)",
            title=dict(
                text=f"<b>{selected_region}</b> — {selected_gran}-Level Demand Forecast",
                font=dict(size=15, color="#aaccff")
            ),
            xaxis=dict(title=xlabel, showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Demand (MW)", showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       titlefont=dict(color="#7a92b8")),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=60, b=0),
            height=430
        )
        st.plotly_chart(fig_f, use_container_width=True)

        # ── Bar chart for coarse granularities ────────────────────────
        if selected_gran in ["Day", "Week", "Month", "Year"]:
            label_col = "label" if "label" in forecast_df.columns else None
            x_vals    = forecast_df.get("label", forecast_df.index.strftime(
                "%d %b" if selected_gran in ["Day","Week"] else "%b %Y"))

            fig_bar = px.bar(
                x=x_vals, y=forecast_df["demand_MW"],
                color=forecast_df["demand_MW"],
                color_continuous_scale="Blues",
                labels={"x": selected_gran, "y": "Avg Demand (MW)"},
                title=f"{selected_gran}ly Avg Demand — {selected_region}",
                text=forecast_df["demand_MW"].round(0).astype(int)
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside",
                                   textfont=dict(color="#aaccff", size=11))
            fig_bar.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)",
                title_font=dict(color="#aaccff"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=50, b=0), height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Data Table & Download ─────────────────────────────────────
        with st.expander("📋 View Forecast Data Table"):
            display_df = forecast_df[["demand_MW", "lower", "upper"]].copy()
            display_df.columns = ["Demand (MW)", "Lower Band (MW)", "Upper Band (MW)"]
            display_df = display_df.round(2)
            st.dataframe(display_df, use_container_width=True, height=250)

            csv_bytes = display_df.to_csv().encode()
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"forecast_{selected_region.lower().replace(' ','_')}_{selected_gran.lower()}.csv",
                mime="text/csv"
            )

        # ── Disclaimer ─────────────────────────────────────────────────
        if is_synthetic:
            st.markdown("""
            <div class="info-box">
            ⚠️ <b>Note:</b> Second and Minute level values are <b>synthesized</b> via cubic
            interpolation of the hourly XGBoost model. They reflect realistic intra-hour
            variability but do <b>not</b> come from raw sub-hourly sensor data.
            </div>""", unsafe_allow_html=True)
        elif is_aggregated:
            st.markdown("""
            <div class="info-box">
            ✅ <b>Note:</b> Day / Week / Month / Year values are computed by <b>aggregating
            multiple hourly predictions</b> from the XGBoost model across the forecast window.
            </div>""", unsafe_allow_html=True)

    else:
        # Idle state — show prompt
        st.markdown("""
        <div style='
            text-align:center; padding:60px 40px;
            background:rgba(0,80,180,0.06);
            border:1px dashed rgba(0,150,255,0.2);
            border-radius:16px; margin-top:20px
        '>
          <div style='font-size:3rem; margin-bottom:16px'>🔮</div>
          <div style='color:#aaccff; font-size:1.1rem; font-weight:600; margin-bottom:8px'>
            Ready to Forecast
          </div>
          <div style='color:#5a7090; font-size:0.85rem'>
            Select a granularity and click <b style="color:#00c6ff">Run Forecast</b> in the sidebar.<br>
            Supports: Second · Minute · Hour · Day · Week · Month · Year
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick-launch granularity cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Available Forecast Granularities</p>',
                    unsafe_allow_html=True)

        gran_info = {
            "Second": ("⏱️", "60 seconds", "Cubic spline interpolation", "#ff9944"),
            "Minute": ("🕐", "60 minutes", "Intra-hour load curves", "#ff9944"),
            "Hour":   ("⏰", "24 hours",   "XGBoost direct prediction", "#00c6ff"),
            "Day":    ("📅", "7 days",     "Hourly aggregation", "#00e676"),
            "Week":   ("📆", "4 weeks",    "Daily aggregation", "#00e676"),
            "Month":  ("🗓️", "12 months",  "Weekly aggregation", "#00e676"),
            "Year":   ("📊", "1 year",     "Monthly pattern", "#00e676"),
        }
        cols = st.columns(7)
        for col, (gran, (icon, default, method, color)) in zip(cols, gran_info.items()):
            col.markdown(f"""
            <div style='
                background:rgba(0,60,120,0.15);
                border:1px solid {color}33;
                border-radius:12px; padding:14px 8px;
                text-align:center; cursor:pointer;
                transition:0.2s;
            '>
              <div style='font-size:1.6rem'>{icon}</div>
              <div style='color:{color}; font-weight:700; font-size:0.85rem; margin:4px 0'>{gran}</div>
              <div style='color:#5a7090; font-size:0.68rem'>{default}</div>
              <div style='color:#3a5070; font-size:0.65rem; margin-top:4px'>{method}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — Model Performance
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">🏆 Model Training Performance</p>',
                unsafe_allow_html=True)

    if meta:
        # Build summary table
        rows = []
        for region, info in meta.items():
            for model_name, m in info.get("metrics", {}).items():
                rows.append({
                    "Region":     region,
                    "Model":      model_name,
                    "MAPE (%)":   round(m.get("mape", 0) * 100, 2),
                    "R² Score":   round(m.get("r2", 0), 4),
                    "RMSE (MW)":  round(m.get("rmse", 0), 1),
                    "Train Rows": info.get("n_train", "-"),
                    "Test Rows":  info.get("n_test", "-"),
                })
        if rows:
            perf_df = pd.DataFrame(rows)

            # Best model per region
            best_df = perf_df.loc[perf_df.groupby("Region")["MAPE (%)"].idxmin()].copy()
            best_df["Grade"] = best_df["MAPE (%)"].apply(
                lambda x: "S (Excellent)" if x < 3 else
                          "A (Great)"     if x < 5 else
                          "B (Good)"      if x < 8 else "C (Fair)"
            )

            st.markdown("#### 📋 Best Model per Region")
            st.dataframe(best_df.set_index("Region"), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # MAPE comparison bar chart
            fig_perf = px.bar(
                perf_df, x="Region", y="MAPE (%)", color="Model", barmode="group",
                color_discrete_sequence=["#00c6ff", "#00e676", "#ff9944"],
                title="MAPE by Region & Model (%)",
            )
            fig_perf.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)",
                title_font=dict(color="#aaccff"),
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis=dict(tickangle=-25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            # R² chart
            fig_r2 = px.bar(
                perf_df, x="Region", y="R² Score", color="Model", barmode="group",
                color_discrete_sequence=["#00c6ff", "#00e676", "#ff9944"],
                title="R² Score by Region & Model",
            )
            fig_r2.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)",
                title_font=dict(color="#aaccff"),
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis=dict(tickangle=-25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.info("No model metadata found. Retrain using `python src/train_multi_granularity.py`.")

    # Feature importance
    st.markdown('<p class="section-header">🔍 Feature Importance</p>', unsafe_allow_html=True)
    model_for_fi = models.get(selected_region)

    try:
        # VotingRegressor or direct XGBoost
        xgb_model = None
        if hasattr(model_for_fi, "estimators_"):
            for est in model_for_fi.estimators_:
                if hasattr(est, "feature_importances_"):
                    xgb_model = est
                    break
        elif hasattr(model_for_fi, "feature_importances_"):
            xgb_model = model_for_fi

        if xgb_model is not None and feature_cols is not None:
            fi = xgb_model.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_cols[:len(fi)], "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=False).head(20)

            fig_fi = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Blues",
                title=f"Top 20 Feature Importances — {selected_region}"
            )
            fig_fi.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,15,40,0.6)",
                title_font=dict(color="#aaccff"),
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=50, b=0), height=500
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
