import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(title="M5 Forecasting API")

# --- 1. CONFIGURATION ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
MODEL_DIR = "models/model_alpha"

MODELS = {}
QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]

# Feature list must match the notebook exactly
FEATURES = [
    'wday', 'month', 'sell_price', 'price_norm',
    'roll_mean_7', 'roll_mean_28',
    'snap_CA', 'snap_TX', 'snap_WI'
]

@app.on_event("startup")
def load_assets():
    for q in QUANTILES:
        path = f"{MODEL_DIR}/model_alpha_{q}.txt"
        if os.path.exists(path):
            MODELS[q] = lgb.Booster(model_file=path)
    print(f"🏁 Startup complete. Loaded {len(MODELS)} models.")

# --- 2. OPTIMIZED DATA FETCHING ---

def get_item_context(item_id: str):
    """Fetches the last available full feature row for the item."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    try:
        # Use Polars to grab the most recent data point for features
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        last_row = df.filter(pl.col("item_id") == item_id).tail(1).collect()
        return last_row.to_dicts()[0] if not last_row.is_empty() else None
    except:
        return None

# --- 3. INFERENCE ENDPOINT ---

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: float = 0.0):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    # 1. Get the context (Static features like store_id, dept_id, etc.)
    ctx = get_item_context(item_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Item data not found.")

    # 2. Setup initial sliding window (Last 28 days of sales)
    # We need history to calculate rolling means dynamically
    sales_history = np.array(get_history(item_id, 56)) 
    
    def run_forecast_loop(seed_sales):
        preds = {q: [] for q in QUANTILES}
        current_window = list(seed_sales)
        
        for i in range(28):
            # RECONSTRUCT FEATURES FOR DAY i
            # In a real MLOps pipeline, you'd join with a calendar here.
            # For the demo, we use the item's average context and update rolling means.
            
            feat_row = [
                ctx['wday'], ctx['month'], ctx['sell_price'], ctx['price_norm'],
                np.mean(current_window[-7:]),  # Dynamic roll_mean_7
                np.mean(current_window[-28:]), # Dynamic roll_mean_28
                ctx['snap_CA'], ctx['snap_TX'], ctx['snap_WI']
            ]
            
            x = np.array(feat_row).reshape(1, -1)
            
            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(x)[0]))
                preds[q].append(p)
            
            # Use the median (0.5) prediction to feed the next step
            current_window.append(preds["0.5"][-1])
            
        return preds

    try:
        # Backtest (t-56 to t-28) and Forecast (t-28 to future)
        bt_preds = run_forecast_loop(sales_history[:28])
        f_preds = run_forecast_loop(sales_history[28:])
        
        actuals = sales_history[28:]
        
        return {
            "item_id": item_id,
            "history": [float(x) for x in actuals],
            "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
            "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
            "metrics": {
                "backtest_accuracy": round(float(np.mean(actuals)), 2) # Example metric
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))