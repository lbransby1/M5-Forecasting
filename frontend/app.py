# frontend/app.py
import streamlit as st
import numpy as np
import warnings

from api_client import fetch_forecast
from components.sidebar import render_sidebar
from components.metrics import render_kpi_cards, render_percentile_panel
from components.charts import plot_probabilistic_forecast, plot_diagnostics
from components.presets import pattern_caption

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] { height: 100%; }
        .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 0.8rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }
        [data-testid="stSidebar"] {
            min-width: 280px;
            max-width: 320px;
        }
        [data-testid="stSidebar"] .block-container { padding-top: 0.8rem !important; }
        .stMetric {
            background-color: #f8f9fb;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #eef0f4;
        }
        div[data-testid="stMetricValue"] { font-size: 1.25rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.8rem; }
        div[data-testid="stSidebar"] button { font-size: 0.8rem; white-space: nowrap; }
        h1, h2 { margin-bottom: 0.15rem !important; }
        [data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
        [data-testid="stTabs"] { margin-top: 0.2rem; }
        iframe { width: 100% !important; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

item_id, selected_store, run_btn, item_meta = render_sidebar()

selection_key = (item_id, selected_store) if item_id and selected_store else None
if selection_key and st.session_state.get("forecast_key") != selection_key:
    st.session_state.pop("forecast_data", None)
    st.session_state.pop("forecast_key", None)

failed_same = st.session_state.get("fetch_failed") == selection_key
should_fetch = bool(selection_key) and (
    run_btn or (st.session_state.get("forecast_key") != selection_key and not failed_same)
)
if should_fetch:
    with st.spinner("Loading quantile forecast..."):
        fetched = fetch_forecast(item_id, selected_store)
        if fetched is None:
            st.session_state["fetch_failed"] = selection_key
        else:
            st.session_state["forecast_data"] = fetched
            st.session_state["forecast_key"] = selection_key
            st.session_state.pop("fetch_failed", None)

forecast_key = (item_id, selected_store) if item_id and selected_store else None
data = (
    st.session_state.get("forecast_data")
    if forecast_key and st.session_state.get("forecast_key") == forecast_key
    else None
)

if data:
    h_sales = np.array(data.get("history", []))
    bt_data = data.get("backtest", {})
    f_data = data.get("forecast", {})

    title_col, meta_col = st.columns([3, 2])
    with title_col:
        st.header(data.get("product_name", item_id))
    with meta_col:
        preset = st.session_state.get("active_preset")
        bits = [f"**{selected_store}**", "28-day horizon"]
        if preset:
            bits.append(f"**{preset}**")
        st.markdown(" · ".join(bits))
        stats = pattern_caption(item_meta)
        if stats:
            st.caption(stats)

    actuals_tail, bt_median, pi_95_upper, pi_95_lower = render_kpi_cards(h_sales, bt_data, f_data)
    render_percentile_panel(f_data)

    tab_forecast, tab_diagnostics = st.tabs(["Probabilistic Forecast", "Model Diagnostics"])
    with tab_forecast:
        daily_col, cumul_col = st.columns(2)
        with daily_col:
            plot_probabilistic_forecast(h_sales, bt_data, f_data, is_cumulative=False, title="Daily units")
        with cumul_col:
            plot_probabilistic_forecast(h_sales, bt_data, f_data, is_cumulative=True, title="Cumulative volume")
    with tab_diagnostics:
        plot_diagnostics(actuals_tail, bt_median, pi_95_upper, pi_95_lower)

else:
    if selection_key:
        st.info("Could not load this forecast. Try another SKU or click **Generate Analytics Report** to retry.")
    st.header("Uncertainty Quantification for Retail Inventory Optimization")
    c1, c2, c3 = st.columns(3)
    c1.markdown("Quantile gradient boosting for localized 28-day demand.")
    c2.markdown("P50–P99 bands for safety-stock planning.")
    c3.markdown("Store-level inference across 10 Walmart regions.")
