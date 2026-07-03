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


def fetch_forecast(item_id: str, store_id: str):
    try:
        res = requests.get(f"{API_URL}/predict/{item_id}?store_id={store_id}", timeout=10)
        
        if res.status_code == 200:
            return res.json()
        elif res.status_code == 503:
            st.error("**Server under maintenance.** Please try again in a few minutes.")
        elif res.status_code == 500:
            st.error("**Internal Server Error.** The backend encountered an issue processing your request.")
        else:
            st.error(f"**Unexpected Error:** Server returned status code {res.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("**Connection refused.** Please check if the backend server is running.")
    except requests.exceptions.Timeout:
        st.error("**Request timed out.** The server is taking too long to respond.")
    except Exception as e:
        st.error(f"An unknown error occurred: {e}")
        
    return None