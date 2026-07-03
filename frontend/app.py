# frontend/app.py
import streamlit as st
import numpy as np
import warnings

from api_client import fetch_forecast
from components.sidebar import render_sidebar
from components.metrics import render_kpi_cards
from components.charts import plot_probabilistic_forecast, plot_diagnostics

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
        .block-container { padding-top: 2rem !important; }
        .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border: 1px solid #eef0f4; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR NAVIGATION ---
item_id, selected_store, run_btn = render_sidebar()

# --- 3. MAIN ANALYTICS ENGINE ---
if run_btn and item_id:
    with st.spinner("Synchronizing recursive quantile forecast..."):
        data = fetch_forecast(item_id, selected_store)
        
        # ADD THIS CHECK:
        if data is None:
            st.stop() 
            
        # PROCEED ONLY IF DATA EXISTS:
        h_sales = np.array(data.get('history', []))
        bt_data = data.get('backtest', {})
        f_data = data.get('forecast', {})

        st.title(f"{data.get('product_name', item_id)}")
        st.info(f"Location: **{selected_store}** | Forecasting Horizon: **28 Days**")

        # Render KPI Cards & get math outputs needed for diagnostics
        actuals_tail, bt_median, pi_95_upper, pi_95_lower = render_kpi_cards(h_sales, bt_data, f_data)

        st.divider()

        # Main Tabs
        tab_forecast, tab_diagnostics = st.tabs(["Probabilistic Forecast", "Model Diagnostics"])

        with tab_forecast:
            # Add the toggle right above the chart
            view_cumulative = st.toggle("Show Cumulative Volume (Smooths daily noise)!!", value=True)
            
            # Pass the toggle state into your new function
            plot_probabilistic_forecast(h_sales, bt_data, f_data, is_cumulative=view_cumulative)

        with tab_diagnostics:
            plot_diagnostics(actuals_tail, bt_median, pi_95_upper, pi_95_lower)

else:
    # Landing Page
    st.title("Uncertainty Quantification for Retail Inventory Optimization - M5 Walmart")
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("Multi-stage Gradient Boosting with recursive lag optimization.")
    c2.markdown("9 distinct quantiles to calculate safety-stock buffers.")
    c3.markdown("Polars backend for sub-second ETL.")