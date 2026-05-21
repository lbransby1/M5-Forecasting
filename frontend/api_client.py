# frontend/api_client.py
import os
import requests
import pandas as pd
import streamlit as st

RAW_API_URL = os.getenv("API_URL", "https://m5-back-production.up.railway.app")
API_URL = RAW_API_URL.rstrip("/")
if not API_URL.startswith("http"): 
    API_URL = f"http://{API_URL}"

@st.cache_data(ttl=600)
def fetch_leaderboard():
    try:
        res = requests.get(f"{API_URL}/leaderboard")
        if res.status_code == 200: 
            payload = res.json()
            return pd.DataFrame(payload.get("data", []))
    except Exception as e: 
        print(f"API Error: {e}")
    return None

def fetch_forecast(item_id: str, store_id: str, current_stock: float):
    try:
        res = requests.get(f"{API_URL}/predict/{item_id}?store_id={store_id}&current_stock={current_stock}")
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
    return None