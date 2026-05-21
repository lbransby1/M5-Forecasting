import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting-Pro")

RAW_API_URL = os.getenv("API_URL", "https://m5-back-production.up.railway.app")
API_URL = RAW_API_URL.rstrip("/")
if not API_URL.startswith("http"): API_URL = f"http://{API_URL}"

# Global CSS for a polished look
st.markdown("""
    <style>
        [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
        .block-container { padding-top: 2rem !important; }
        .stMetric { background-color: #f8f9fb; padding: 15px; border-radius: 10px; border: 1px solid #eef0f4; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ASSET FETCHING ---
@st.cache_data(ttl=600)
def fetch_leaderboard():
    try:
        res = requests.get(f"{API_URL}/leaderboard")
        if res.status_code == 200: 
            # Extract the 'data' list from the JSON payload
            payload = res.json()
            return pd.DataFrame(payload.get("data", []))
    except Exception as e: 
        print(f"API Error: {e}")
        pass
    return None

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Inventory Ops")
df_items = fetch_leaderboard()
item_id = None
selected_store = None

if df_items is not None and not df_items.empty:
    # Location Filter
    # --- SAFE STORE EXTRACTION ---
    # Check if the backend successfully provided the store_id column
    if 'store_id' in df_items.columns:
        # Drop any nulls just in case, then get unique stores
        valid_stores = df_items['store_id'].dropna().unique()
        stores = sorted(list(valid_stores)) if len(valid_stores) > 0 else ["CA_1"]
    else:
        # Fallback to the standard 10 stores of the M5 dataset if the column is missing
        stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
        # Add a dummy column so subsequent code that filters by store_id doesn't crash
        df_items['store_id'] = "CA_1"

    selected_store = st.sidebar.selectbox("Select Store Location", stores)
    df_store = df_items[df_items['store_id'] == selected_store]

    # Smart Search
    search = st.sidebar.text_input("Search Product Name/ID", "").lower()
    filtered = df_store
    if search:
        filtered = filtered[
            (filtered['product_name'].str.lower().str.contains(search)) | 
            (filtered['item_id'].str.lower().str.contains(search))
        ]

    # Item Selection
    item_options = filtered.set_index('item_id')['product_name'].to_dict()
    item_id = st.sidebar.selectbox(
        "Target SKU", 
        options=list(item_options.keys()),
        format_func=lambda x: item_options[x]
    )

st.sidebar.divider()
current_stock = st.sidebar.number_input("Warehouse Stock Level", value=20, help="Initial inventory for risk simulation")
run_btn = st.sidebar.button("Generate Analytics Report", type="primary", disabled=(item_id is None))

# --- 4. MAIN ANALYTICS ENGINE ---
if run_btn and item_id:
    with st.spinner("Synchronizing recursive quantile forecast..."):
        try:
            # API Request
            res = requests.get(f"{API_URL}/predict/{item_id}?store_id={selected_store}&current_stock={current_stock}")
            data = res.json()
            
            # Data Unpacking
            h_sales = np.array(data.get('history', []))
            bt_data = data.get('backtest', {})
            f_data = data.get('forecast', {})
            
            # --- DATA SCIENCE METRICS CALCULATIONS ---
            # 1. Backtest MAE (Mean Absolute Error on the last 28 days)
            actuals_tail = h_sales[-28:]
            bt_median = np.array(bt_data.get('0.5', [0]*28))
            mae = np.mean(np.abs(actuals_tail - bt_median))
            
            # 2. Uncertainty (Mean Width of the 95% Prediction Interval)
            pi_95_upper = np.array(f_data.get('0.975', [0]*28))
            pi_95_lower = np.array(f_data.get('0.025', [0]*28))
            uncertainty_width = np.mean(pi_95_upper - pi_95_lower)
            
            # 3. Capture Rate (Calibration check: % of actuals inside 95% ribbon)
            bt_95_upper = np.array(bt_data.get('0.975', [0]*28))
            bt_95_lower = np.array(bt_data.get('0.025', [0]*28))
            within_bounds = np.sum((actuals_tail >= bt_95_lower) & (actuals_tail <= bt_95_upper))
            capture_rate = (within_bounds / 28) * 100

            # UI Header
            st.title(f"📈 {data.get('product_name', item_id)}")
            st.info(f"Location: **{selected_store}** | Forecasting Horizon: **28 Days**")

            # Dashboard Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Capture Rate (95% PI)", f"{round(capture_rate, 1)}%", 
                      help="Calibration Check: Should be near 95%")
            m2.metric("Backtest MAE", f"{round(mae, 2)}", delta_color="inverse", 
                      help="Mean Absolute Error of the Median prediction")
            m3.metric("Uncertainty Width", f"{round(uncertainty_width, 1)}", 
                      help="Average width of 95% Forecast Ribbon (Volatility Indicator)")
            m4.metric("Avg History", f"{round(np.mean(h_sales), 2)} units")

            st.divider()

            # Main Tabs
            tab_forecast, tab_diagnostics = st.tabs(["📊 Probabilistic Forecast", "🧠 Model Diagnostics"])

            with tab_forecast:
                # Time axis setup
                history_len = len(h_sales)
                h_days = list(range(-history_len + 1, 1))
                bt_days = list(range(-27, 1))
                f_days = list(range(1, 29))

                fig = go.Figure()

                def add_fan(figure, d, days, color_base, name_prefix):
                    intervals = [('0.005', '0.995', 0.1, '99% PI'), ('0.025', '0.975', 0.2, '95% PI'),
                                 ('0.165', '0.835', 0.3, '67% PI'), ('0.25', '0.75', 0.4, '50% PI')]
                    for low, high, alpha, label in intervals:
                        if low in d and high in d:
                            figure.add_trace(go.Scatter(
                                x=days + days[::-1], y=d[high] + d[low][::-1],
                                fill='toself', fillcolor=f'rgba({color_base}, {alpha})',
                                line=dict(color='rgba(255,255,255,0)'), name=f"{name_prefix} {label}", hoverinfo="skip"
                            ))
                    if '0.5' in d:
                        figure.add_trace(go.Scatter(x=days, y=d['0.5'], name=f"{name_prefix} Median", 
                                                  line=dict(color=f'rgb({color_base})', width=3)))

                # Actual Sales
                fig.add_trace(go.Bar(x=h_days, y=h_sales, name="Actual Sales", marker_color='rgba(150,150,150,0.3)'))
                # Overlays
                add_fan(fig, bt_data, bt_days, "220, 50, 50", "Backtest")
                add_fan(fig, f_data, f_days, "0, 100, 255", "Future")

                fig.update_layout(
                    height=600, template="plotly_white", hovermode="x unified",
                    xaxis=dict(title="Timeline (Days)", range=[-history_len, 30]),
                    yaxis=dict(title="Units Sold"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_diagnostics:
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("Residual Distribution")
                    residuals = actuals_tail - bt_median
                    res_fig = go.Figure(data=[go.Histogram(x=residuals, nbinsx=15, marker_color='red')])
                    res_fig.update_layout(title="Prediction Residuals (Actual - Median)", template="plotly_white")
                    st.plotly_chart(res_fig, use_container_width=True)
                
                with col_right:
                    st.subheader("Uncertainty Scaling")
                    # Show how 95% PI expands into the future
                    width_f = pi_95_upper - pi_95_lower
                    width_fig = go.Figure(data=[go.Scatter(x=f_days, y=width_f, mode='lines+markers', line_color='blue')])
                    width_fig.update_layout(title="Forecast Variance (95% PI Width Over Time)", template="plotly_white")
                    st.plotly_chart(width_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Inference Pipeline Error: {e}")
            st.exception(e)
else:
    # Landing Page
    st.title("M5 Smart-Supply Engine")
    st.write("Professional Quantile Forecasting for Retail Inventory Optimization.")
    
    # Feature highlights
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🎯 Accuracy\nMulti-stage Gradient Boosting with recursive lag optimization.")
    c2.markdown("### 🛡️ Risk Management\n9 distinct quantiles to calculate safety-stock buffers.")
    c3.markdown("### ⚡ Low Latency\nRust-backed Polars backend for sub-second ETL.")