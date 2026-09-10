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

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
        .block-container { padding-top: 2rem !important; }
        .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border: 1px solid #eef0f4; }
        div[data-testid="stSidebar"] button { font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR NAVIGATION ---
item_id, selected_store, run_btn, item_meta = render_sidebar()

selection_key = (item_id, selected_store) if item_id and selected_store else None
if selection_key and st.session_state.get("forecast_key") != selection_key:
    st.session_state.pop("forecast_data", None)
    st.session_state.pop("forecast_key", None)

# --- 3. MAIN ANALYTICS ENGINE ---
# Auto-load the high-velocity default (and any later SKU/store change). Persist across tab toggles.
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

    st.title(f"{data.get('product_name', item_id)}")
    preset = st.session_state.get("active_preset")
    location = f"Location: **{selected_store}** | Horizon: **28 days**"
    if preset:
        location += f" | Pattern: **{preset}**"
    st.info(location)
    stats = pattern_caption(item_meta)
    if stats:
        st.caption(stats)

    actuals_tail, bt_median, pi_95_upper, pi_95_lower = render_kpi_cards(h_sales, bt_data, f_data)

    st.divider()
    render_percentile_panel(f_data)
    st.divider()

    tab_forecast, tab_diagnostics = st.tabs(["Probabilistic Forecast", "Model Diagnostics"])

    with tab_forecast:
        st.subheader("Daily units")
        plot_probabilistic_forecast(h_sales, bt_data, f_data, is_cumulative=False)
        st.subheader("Cumulative volume")
        plot_probabilistic_forecast(h_sales, bt_data, f_data, is_cumulative=True)

    with tab_diagnostics:
        plot_diagnostics(actuals_tail, bt_median, pi_95_upper, pi_95_lower)

else:
    if selection_key:
        st.info("Could not load this forecast. Click **Generate Analytics Report** to retry.")
    st.title("Uncertainty Quantification for Retail Inventory Optimization - M5 Walmart")

    c1, c2, c3 = st.columns(3)
    c1.markdown("Multi-stage Gradient Boosting with recursive lag optimization.")
    c2.markdown("9 distinct quantiles to calculate safety-stock buffers.")
    c3.markdown("Polars backend for sub-second ETL.")
