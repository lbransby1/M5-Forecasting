# frontend/components/sidebar.py
import streamlit as st
import pandas as pd
from api_client import fetch_leaderboard

M5_STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]

def render_sidebar():
    st.sidebar.title("Inventory")
    df_items = fetch_leaderboard()
    
    item_id = None
    selected_store = None
    
    if df_items is not None and not df_items.empty:
        if "store_id" not in df_items.columns:
            expanded = []
            for _, row in df_items.iterrows():
                for store in M5_STORES:
                    expanded.append({**row.to_dict(), "store_id": store})
            df_items = pd.DataFrame(expanded)

        valid_stores = df_items["store_id"].dropna().unique()
        stores = sorted(list(valid_stores)) if len(valid_stores) > 0 else M5_STORES

        selected_store = st.sidebar.selectbox("Select Store Location", stores)
        df_store = df_items[df_items["store_id"] == selected_store]

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
    run_btn = st.sidebar.button("Generate Analytics Report", type="primary", disabled=(item_id is None))
    
    return item_id, selected_store, run_btn