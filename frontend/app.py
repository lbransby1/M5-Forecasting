import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import toml
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 1. SET PAGE CONFIG ---
#config = toml.load("config.toml")
st.set_page_config(layout="wide", page_icon="📈", page_title="M5-Forecasting.ino")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- 2. FORCE SIDEBAR WIDTH ---

# --- 3. AGGRESSIVE UI TIGHTENING & SCROLLBAR REMOVAL ---
st.markdown(
    """
    <style>
        /* 1. Sidebar Width & Removal of the 'Chevron' (Collapse Button) */
        [data-testid="stSidebar"] {
            min-width: 350px;
            max-width: 350px;
        }
        
        /* Targets the button specifically to remove it and its space */
        [data-testid="stSidebarCollapsedControl"], 
        button[kind="header"] {
            display: none !important;
        }

        /* 2. Remove whitespace at top of Sidebar */
        [data-testid="stSidebarUserContent"] {
            padding-top: 0rem !important;
        }

        /* 3. KILL THE SIDEBAR SCROLLBAR */
        [data-testid="stSidebar"] > div:first-child {
            overflow-y: hidden !important; /* Forces scrollbar to disappear */
            overflow-x: hidden !important;
        }

        /* For Firefox */
        [data-testid="stSidebar"] {
            scrollbar-width: none;
        }

        /* For Chrome, Safari, and Edge */
        [data-testid="stSidebar"]::-webkit-scrollbar {
            display: none;
        }

        /* 4. Reduce gap at the top of the Main Page */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem;
        }

        /* 5. Remove the invisible header shelf completely */
        header {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar: Global Filtering & Search ---
st.sidebar.header("Inventory Explorer")

@st.cache_data(ttl=600)
def fetch_leaderboard():
    try:
        res = requests.get(f"{API_URL}/leaderboard")
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            df.columns = df.columns.str.strip()
            return df
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")
        return None
    return None

df_items = fetch_leaderboard()
item_id = None

if df_items is not None and not df_items.empty:
    if 'dept_id' in df_items.columns:
        depts = ["All"] + sorted(list(df_items['dept_id'].unique()))
        selected_dept = st.sidebar.selectbox("Filter Department", depts)
        if selected_dept != "All":
            df_items = df_items[df_items['dept_id'] == selected_dept]

    search_query = st.sidebar.text_input("Search Product Name", "").lower()
    name_col = 'product_name' if 'product_name' in df_items.columns else 'item_id'
    if search_query:
        df_items = df_items[df_items[name_col].str.lower().str.contains(search_query)]

    available_metrics = [m for m in ["Popularity", "Volatility", "Sparsity"] if m in df_items.columns]
    if available_metrics:
        sort_metric = st.sidebar.selectbox("Sort Items By", available_metrics)
        sort_order = st.sidebar.toggle("Ascending Order", value=False)
        df_items = df_items.sort_values(by=sort_metric, ascending=sort_order)
    else:
        sort_metric = None

    if not df_items.empty:
        item_display_list = []
        for _, row in df_items.iterrows():
            name = row.get('product_name', row.get('item_id', 'Unknown'))
            metric_val = row.get(sort_metric, "N/A") if sort_metric else "N/A"
            item_display_list.append(f"{name} ({sort_metric}: {metric_val})")

        selected_display = st.sidebar.selectbox("Select Product", options=item_display_list)
        idx = item_display_list.index(selected_display)
        item_id = df_items.iloc[idx]['item_id']
    else:
        st.sidebar.warning("No items match filters.")
else:
    st.sidebar.error("Could not fetch leaderboard. Please ensure backend is running.")

st.sidebar.divider()
current_stock = st.sidebar.number_input("Warehouse Stock Level", value=20)
run_btn = st.sidebar.button("Run Analytics Report", type="primary", disabled=(item_id is None))

# --- MAIN PAGE LOGIC ---
if run_btn and item_id:
    # --- SHOW ANALYSIS REPORT ---
    st.title("M5 Walmart Inventory Management System")
    st.caption("Recursive LightGBM 9-Quantile Forecasting for stocking decisions")
    st.divider()

    with st.spinner("Analyzing accuracy and forecasting..."):
        try:
            res = requests.get(f"{API_URL}/predict/{item_id}?current_stock={current_stock}")
            data = res.json()
            
            metrics = data.get('metrics', {})
            bt_data = data.get('backtest', {})
            f_data = data.get('forecast', {})
            h_sales = data.get('history', [])
            risk = data.get('risk_assessment', {"level": "Unknown", "action": "Check Backend"})

            # --- 1. VALIDATION METRICS ---
            st.subheader(f"{data.get('product_name', item_id)}")
            v1, v2, v3 = st.columns(3)
            v1.metric("WSPL Score", metrics.get('wspl', 'N/A'), help="Weighted Scaled Pinball Loss.")
            v2.metric("Interval Capture", f"{metrics.get('backtest_accuracy', 'N/A')}%", help="Backtest accuracy.")
            v3.metric("Model Reliability", "High" if metrics.get('backtest_accuracy', 0) >= 80 else "Moderate")

            # --- 2. DUAL-FAN VISUALIZATION ---
            fig = go.Figure()
            h_days = list(range(-len(h_sales)+1, 1))
            bt_days = list(range(-len(h_sales)+1, 1))
            f_days = list(range(1, 29))

            def add_fan(figure, d, days, color_base, name_prefix):
                intervals = [('0.025', '0.975', 0.1, '95% PI'), ('0.05', '0.95', 0.15, '90% PI'),
                             ('0.1', '0.9', 0.2, '80% PI'), ('0.25', '0.75', 0.3, '50% PI')]
                for low, high, alpha, label in intervals:
                    if low in d and high in d:
                        figure.add_trace(go.Scatter(
                            x=days + days[::-1], y=d[high] + d[low][::-1],
                            fill='toself', fillcolor=f'rgba({color_base}, {alpha})',
                            line=dict(color='rgba(255,255,255,0)'), name=f"{name_prefix} {label}", hoverinfo="skip"
                        ))
                if '0.5' in d:
                    figure.add_trace(go.Scatter(x=days, y=d['0.5'], name=f"{name_prefix} Median", line=dict(color=f'rgb({color_base})', width=4)))

            fig.add_trace(go.Bar(x=h_days, y=h_sales, name="Actual Sales", marker_color='rgba(100,100,100,0.25)'))
            add_fan(fig, bt_data, bt_days, "220, 50, 50", "Backtest")
            add_fan(fig, f_data, f_days, "0, 100, 255", "Future")
            fig.update_layout(height=550, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(range=[-28, 29])
            st.plotly_chart(fig, width="stretch", theme=None)

            # --- 3. STOCKING RISK CARD ---
            st.info(f"**Stocking Analysis:** {risk.get('level')} — {risk.get('action')}")

            # --- 4. ADVANCED DIAGNOSTICS & GLOSSARY ---
            st.divider()
            diag_col, gloss_col = st.columns([2, 1])

            with diag_col:
                st.markdown("### Statistical Diagnostics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Interval Width (MPIW)", f"{metrics.get('mpiw', 'N/A')} units")
                c2.metric("Intermittency", f"{metrics.get('zero_pct', 'N/A')}%")
                cv = metrics.get('cv', 0)
                c3.metric("Volatility (CV)", cv, delta="High" if cv > 1.0 else "Low", delta_color="inverse")

                st.markdown("---")
                d1, d2 = st.columns(2)
                d1.metric("Weekly Seasonality (ACF)", metrics.get('seasonality_7d', 'N/A'))
                capture = metrics.get('backtest_accuracy', 0)
                d2.metric("Calibration State", "Overconfident" if capture < 85 else "Underconfident" if capture > 98 else "Calibrated")

            with gloss_col:
                st.markdown("### Glossary")
                with st.expander("What do these mean?", expanded=True):
                    st.write("**WSPL:** Error score for probability. Lower = better distribution fit.")
                    st.write("**Interval Capture:** How often actual sales fell within our 'Ribbon'. Target is 95%.")
                    st.write("**MPIW:** The 'Thickness' of the ribbon. Narrower ribbons mean higher model certainty.")
                    st.write("**Volatility (CV):** Consistency of sales. Above 1.0 means demand is 'Lumpy'.")
                    st.write("**Intermittency:** Percentage of days with zero sales.")
                    st.write("**ACF (Seasonality):** How much today's sales depend on sales exactly 7 days ago.")

        except Exception as e:
            st.error(f"Visualization Error: {e}")

else:
    # --- SHOW LANDING PAGE (Default State) ---
    st.title("M5 Walmart Inventory Management System")
    st.subheader("Recursive LightGBM 9-Quantile Forecasting")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Strategic Inventory Management
        Welcome to the M5 Forecasting Engine. This application uses probabilistic machine learning to 
        move beyond point-forecasts, allowing you to visualize uncertainty and stock-out risks.
        
        **How to use this tool:**
        1. **Select a Product:** Use the sidebar to filter by department or search for an Item ID.
        2. **Configure Stock:** Enter your current warehouse inventory level.
        3. **Generate Report:** Click the primary button to see the 28-day future "Fan Chart".
        """)
        
        if df_items is not None:
            st.markdown("#### High-Priority Items")
            st.dataframe(df_items.head(10), hide_index=True, width="stretch")

    with col2:
        st.info("### Quick Tips")
        st.write("**95% PI:** The widest ribbon represents our 95% certainty interval.")
        st.write("**Calibration:** We target a 95% capture rate for reliable safety-stock planning.")
        st.write("**Sorting:** Sort by 'Volatility' to identify items that need the most buffer stock.")

    st.divider()
    st.caption("Powered by LightGBM & Streamlit. Data sourced from Walmart M5 Competition.")