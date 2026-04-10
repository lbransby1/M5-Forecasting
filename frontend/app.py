import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting")

RAW_API_URL = os.getenv("API_URL", "https://m5-back-production.up.railway.app")
API_URL = RAW_API_URL.rstrip("/")
if not API_URL.startswith("http"): API_URL = f"http://{API_URL}"

st.markdown("""<style>
    [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
    .block-container { padding-top: 1rem !important; }
</style>""", unsafe_allow_html=True)

st.sidebar.header("Inventory Explorer")

@st.cache_data(ttl=600)
def fetch_leaderboard():
    try:
        res = requests.get(f"{API_URL}/leaderboard")
        if res.status_code == 200: return pd.DataFrame(res.json())
    except: pass
    return None

df_items = fetch_leaderboard()
item_id = None
selected_store = None

if df_items is not None and not df_items.empty:
    # 1. STORE FILTER
    stores = sorted(list(df_items['store_id'].unique()))
    selected_store = st.sidebar.selectbox("Select Store Location", stores)
    df_store = df_items[df_items['store_id'] == selected_store]

    # 2. SEARCH BY PRODUCT NAME
    search = st.sidebar.text_input("Search Product (e.g., Stickers)", "").lower()
    
    filtered = df_store
    if search:
        filtered = filtered[
            (filtered['product_name'].str.lower().str.contains(search)) | 
            (filtered['item_id'].str.lower().str.contains(search))
        ]

    # Map Name to ID for the selector
    item_options = filtered.set_index('item_id')['product_name'].to_dict()
    item_id = st.sidebar.selectbox(
        "Select Product", 
        options=list(item_options.keys()),
        format_func=lambda x: item_options[x]
    )

st.sidebar.divider()
current_stock = st.sidebar.number_input("Warehouse Stock Level", value=20)
run_btn = st.sidebar.button("Run Analytics Report", type="primary", disabled=(item_id is None))

if run_btn and item_id:
    with st.spinner("Fetching prediction data..."):
        try:
            res = requests.get(f"{API_URL}/predict/{item_id}?store_id={selected_store}&current_stock={current_stock}")
            data = res.json()
            
            st.title(f"Inventory Report: {data.get('product_name', item_id)}")
            st.caption(f"Location: {selected_store} | Recursive Quantile Forecast")

            h_sales = data.get('history', [])
            bt_data = data.get('backtest', {})
            f_data = data.get('forecast', {})

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

            fig.add_trace(go.Bar(x=h_days, y=h_sales, name="Actual Sales", marker_color='rgba(100,100,100,0.2)'))
            add_fan(fig, bt_data, bt_days, "220, 50, 50", "Backtest")
            add_fan(fig, f_data, f_days, "0, 100, 255", "Future")

            fig.update_layout(
                height=550, template="plotly_white", hovermode="x unified",
                xaxis=dict(title="Days Offset from Today", range=[-history_len, 30]),
                yaxis=dict(title="Units Sold"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.title("M5 Walmart Inventory Management System")
    st.info("Please select a Store and Product in the sidebar to begin.")