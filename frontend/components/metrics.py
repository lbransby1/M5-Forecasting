# frontend/components/metrics.py
import streamlit as st
import numpy as np

def render_kpi_cards(h_sales, bt_data, f_data):
    actuals_tail = h_sales[-28:]
    bt_median = np.array(bt_data.get('0.5', [0]*28))
    
    # 1. Backtest MAE
    mae = np.mean(np.abs(actuals_tail - bt_median))
    
    # 2. Uncertainty Width
    pi_95_upper = np.array(f_data.get('0.975', [0]*28))
    pi_95_lower = np.array(f_data.get('0.025', [0]*28))
    uncertainty_width = np.mean(pi_95_upper - pi_95_lower)
    
    # 3. Capture Rate
    bt_95_upper = np.array(bt_data.get('0.975', [0]*28))
    bt_95_lower = np.array(bt_data.get('0.025', [0]*28))
    within_bounds = np.sum((actuals_tail >= bt_95_lower) & (actuals_tail <= bt_95_upper))
    capture_rate = (within_bounds / 28) * 100

    # Render UI
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capture Rate (95% PI)", f"{round(capture_rate, 1)}%", help="Calibration Check: Should be near 95%")
    m2.metric("Backtest MAE", f"{round(mae, 2)}", delta_color="inverse", help="Mean Absolute Error of the Median prediction")
    m3.metric("Uncertainty Width", f"{round(uncertainty_width, 1)}", help="Average width of 95% Forecast Ribbon")
    m4.metric("Avg History", f"{round(np.mean(h_sales), 2)} units")
    
    return actuals_tail, bt_median, pi_95_upper, pi_95_lower