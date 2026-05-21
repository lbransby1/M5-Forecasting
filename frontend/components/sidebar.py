# frontend/components/sidebar.py
import streamlit as st
from api_client import fetch_leaderboard

def render_sidebar():
    st.sidebar.title("🛠️ Inventory Ops")
    df_items = fetch_leaderboard()
    
    item_id = None
    selected_store = None
    
    if df_items is not None and not df_items.empty:
        # Safe store extraction
        if 'store_id' in df_items.columns:
            valid_stores = df_items['store_id'].dropna().unique()
            stores = sorted(list(valid_stores)) if len(valid_stores) > 0 else ["CA_1"]
        else:
            stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
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
        if item_options:
            item_id = st.sidebar.selectbox(
                "Target SKU", 
                options=list(item_options.keys()),
                format_func=lambda x: item_options[x]
            )

    st.sidebar.divider()
    current_stock = st.sidebar.number_input("Warehouse Stock Level", value=20, help="Initial inventory for risk simulation")
    run_btn = st.sidebar.button("Generate Analytics Report", type="primary", disabled=(item_id is None))
    
    return item_id, selected_store, current_stock, run_btn