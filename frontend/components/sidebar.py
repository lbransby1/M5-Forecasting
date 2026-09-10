# frontend/components/sidebar.py
import streamlit as st
import pandas as pd
from api_client import fetch_leaderboard
from components.presets import PRESET_ORDER, pattern_caption, resolve_presets

M5_STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
DEFAULT_STORE = "CA_3"


def _apply_preset(item_id: str, preset_name: str):
    st.session_state["sku_select"] = item_id
    st.session_state["product_search"] = ""
    st.session_state["active_preset"] = preset_name
    st.session_state.pop("fetch_failed", None)


def render_sidebar():
    st.sidebar.title("Inventory")
    df_items = fetch_leaderboard()
    presets = resolve_presets(df_items)

    item_id = None
    selected_store = None
    item_meta = None

    if df_items is not None and not df_items.empty:
        if "store_id" not in df_items.columns:
            expanded = []
            for _, row in df_items.iterrows():
                for store in M5_STORES:
                    expanded.append({**row.to_dict(), "store_id": store})
            df_items = pd.DataFrame(expanded)

        valid_stores = df_items["store_id"].dropna().unique()
        stores = sorted(list(valid_stores)) if len(valid_stores) > 0 else M5_STORES

        if "store_select" not in st.session_state:
            st.session_state["store_select"] = DEFAULT_STORE if DEFAULT_STORE in stores else stores[0]
        elif st.session_state["store_select"] not in stores:
            st.session_state["store_select"] = stores[0]

        default_sku = presets["High Velocity"]["item_id"]
        if "sku_select" not in st.session_state:
            st.session_state["sku_select"] = default_sku
            st.session_state["active_preset"] = "High Velocity"

        preset_ids = {presets[name]["item_id"]: name for name in PRESET_ORDER}
        current_sku = st.session_state.get("sku_select")
        st.session_state["active_preset"] = preset_ids.get(current_sku)

        st.sidebar.caption("Demand pattern")
        for names in (PRESET_ORDER[:2], PRESET_ORDER[2:]):
            cols = st.sidebar.columns(len(names))
            for col, name in zip(cols, names):
                is_active = st.session_state.get("active_preset") == name
                col.button(
                    name,
                    key=f"preset_{name.replace(' ', '_').lower()}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    on_click=_apply_preset,
                    args=(presets[name]["item_id"], name),
                    help=presets[name]["blurb"],
                )

        active = st.session_state.get("active_preset")
        if active in presets:
            st.sidebar.caption(presets[active]["blurb"])

        selected_store = st.sidebar.selectbox("Select Store Location", stores, key="store_select")
        df_store = df_items[df_items["store_id"] == selected_store]

        search = st.sidebar.text_input("Search Product Name/ID", key="product_search")
        filtered = df_store
        if search:
            query = search.lower()
            filtered = filtered[
                (filtered["product_name"].astype(str).str.lower().str.contains(query, na=False))
                | (filtered["item_id"].astype(str).str.lower().str.contains(query, na=False))
            ]

        item_options = filtered.set_index("item_id")["product_name"].to_dict()
        options = list(item_options.keys())
        if options:
            if st.session_state.get("sku_select") not in options:
                st.session_state["sku_select"] = options[0]
            item_id = st.sidebar.selectbox(
                "Target SKU",
                options=options,
                format_func=lambda x: item_options[x],
                key="sku_select",
            )
            match = filtered[filtered["item_id"] == item_id]
            if not match.empty:
                item_meta = match.iloc[0].to_dict()

            stats = pattern_caption(item_meta)
            if stats:
                st.sidebar.caption(stats)

    st.sidebar.divider()
    run_btn = st.sidebar.button("Generate Analytics Report", type="primary", disabled=(item_id is None))

    return item_id, selected_store, run_btn, item_meta
