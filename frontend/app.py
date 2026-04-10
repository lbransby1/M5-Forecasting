import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. SET PAGE CONFIG ---
st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting")

RAW_API_URL = os.getenv("API_URL", "https://m5-back-production.up.railway.app")
API_URL = RAW_API_URL.rstrip("/")
if not API_URL.startswith("http"):
    API_URL = f"http://{API_URL}"

# --- 2. UI STYLING ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
        [data-testid="stSidebarCollapsedControl"], button[kind="header"] { display: none !important; }
        [data-testid="stSidebarUserContent"] { padding-top: 0rem !important; }
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem; }
        header { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. SIDEBAR LOGIC ---
st.sidebar.header("Inventory Explorer")

@st.cache_data(ttl=600)
def fetch_leaderboard():
    try:
        res = requests.get(f"{API_URL}/leaderboard")
        if res.status_code == 200:
            return pd.DataFrame(res.json())
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")
    return None

df_items = fetch_leaderboard()
item_id = None

if df_items is not None and not df_items.empty:
    # Department Filter
    depts = ["All"] + sorted(list(df_items['dept_id'].unique()))
    selected_dept = st.sidebar.selectbox("Filter Department", depts)
    if selected_dept != "All":
        df_items = df_items[df_items['dept_id'] == selected_dept]

    # Search & Sort
    search_query = st.sidebar.text_input("Search Product Name", "").lower()
    if search_query:
        df_items = df_items[df_items['item_id'].str.lower().str.contains(search_query)]

    # Item Selection
    item_display_list = df_items['item_id'].tolist()
    item_id = st.sidebar.selectbox("Select Product", options=item_display_list)

st.sidebar.divider()
current_stock = st.sidebar.number_input("Warehouse Stock Level", value=20)
run_btn = st.sidebar.button("Run Analytics Report", type="primary", disabled=(item_id is None))

# --- 4. MAIN PAGE & VISUALIZATION ---
if run_btn and item_id:
    st.title("M5 Walmart Inventory Management System")
    st.caption("Recursive LightGBM 9-Quantile Forecasting")
    st.divider()

    with st.spinner("Synchronizing 84-day history and forecast..."):
        try:
            res = requests.get(f"{API_URL}/predict/{item_id}?current_stock={current_stock}")
            data = res.json()
            
            h_sales = data.get('history', [])
            bt_data = data.get('backtest', {})
            f_data = data.get('forecast', {})
            metrics = data.get('metrics', {})

            # --- DYNAMIC TIME AXIS ALIGNMENT ---
            # History is 84 days: Days -83 to 0
            history_len = len(h_sales)
            h_days = list(range(-history_len + 1, 1))
            
            # Backtest covers the last 28 days of history: Days -27 to 0
            bt_days = list(range(-27, 1))
            
            # Future starts tomorrow: Days 1 to 28
            f_days = list(range(1, 29))

            fig = go.Figure()

            # Helper for Quantile Fans
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

            # 1. Historical Bars (Full 84 days)
            fig.add_trace(go.Bar(x=h_days, y=h_sales, name="Actual Sales", marker_color='rgba(100,100,100,0.2)'))
            
            # 2. Backtest Overlay (Last 28 days)
            add_fan(fig, bt_data, bt_days, "220, 50, 50", "Backtest")
            
            # 3. Future Forecast (Next 28 days)
            add_fan(fig, f_data, f_days, "0, 100, 255", "Future")

            fig.update_layout(
                height=550, template="plotly_white", hovermode="x unified",
                xaxis=dict(title="Days (Relative to Today)", range=[-history_len, 30]),
                yaxis=dict(title="Units Sold"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Metrics Section
            m1, m2, m3 = st.columns(3)
            m1.metric("Historical Avg", f"{round(np.mean(h_sales), 2)} units")
            m2.metric("Forecast Median", f"{round(np.mean(f_data.get('0.5', [0])), 2)} units")
            m3.metric("System Status", "Live", delta="Calibrated")

        except Exception as e:
            st.error(f"Visualization Error: {e}")
else:
    st.title("M5 Walmart Inventory Management System")
    st.info("Select an item from the sidebar to view the 84-day history and probabilistic forecast.")