import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(title="M5 Forecasting API")

# --- 1. MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIGURATION & CALENDAR ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
MODEL_DIR = "models/model_alpha"

# Load Calendar globally for fast lookups
CALENDAR = pd.read_csv("backend/data/raw/calendar.csv")
CALENDAR['d_num'] = CALENDAR['d'].str.replace('d_', '').astype(int)

MODELS = {}
QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]

@app.on_event("startup")
def load_assets():
    print(f"🚀 Loading Alpha Models from {MODEL_DIR}...")
    for q in QUANTILES:
        path = f"{MODEL_DIR}/model_alpha_{q}.txt"
        if os.path.exists(path):
            MODELS[q] = lgb.Booster(model_file=path)
    print(f"🏁 Startup complete. Loaded {len(MODELS)} models.")

# --- 3. HELPERS ---

def get_history(item_id, count=56):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return [0.0] * count
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = df.filter(pl.col("item_id") == item_id).tail(count).collect()
        return result["sales"].to_list() if not result.is_empty() else [0.0] * count
    except:
        return [0.0] * count

def get_item_context(item_id: str):
    if not os.path.exists(PROCESSED_DATA_PATH): return None
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        last_row = df.filter(pl.col("item_id") == item_id).tail(1).collect()
        return last_row.to_dicts()[0] if not last_row.is_empty() else None
    except: return None

# --- 4. PREDICTION ---

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: float = 0.0):
    if not MODELS: raise HTTPException(status_code=503, detail="Models not loaded.")
    
    ctx = get_item_context(item_id)
    if not ctx: raise HTTPException(status_code=404, detail="Item not found.")

    sales_history = np.array(get_history(item_id, 56)) 
    if len(sales_history) < 56:
        sales_history = np.pad(sales_history, (56 - len(sales_history), 0), 'constant')

    # Get the current day 'd' from data (e.g., 1941)
    current_d = int(ctx.get('d', 1941))

    def run_forecast_loop(seed_sales, start_d):
        preds = {q: [] for q in QUANTILES}
        current_window = list(seed_sales)
        
        for i in range(28):
            target_day = start_d + i
            day_info = CALENDAR[CALENDAR['d_num'] == target_day].iloc[0]
            
            feat_row = [
                ctx.get('item_id'), ctx.get('dept_id'), ctx.get('cat_id'), 
                ctx.get('store_id'), ctx.get('state_id'),
                day_info['wday'], day_info['month'], 
                ctx.get('sell_price', 0), ctx.get('price_norm', 0),
                np.mean(current_window[-7:]),  
                np.mean(current_window[-28:]), 
                day_info['snap_CA'], day_info['snap_TX'], day_info['snap_WI']
            ]
            
            df_exec = pd.DataFrame([feat_row], columns=[
                'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                'wday', 'month', 'sell_price', 'price_norm', 
                'roll_mean_7', 'roll_mean_28', 'snap_CA', 'snap_TX', 'snap_WI'
            ])
            
            for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
                df_exec[col] = df_exec[col].astype('category')

            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(df_exec)[0]))
                preds[q].append(p)
            
            current_window.append(preds["0.5"][-1])
        return preds

    try:
        # ✅ FIX: Passing start_d to functions
        # Backtest starts 28 days ago
        bt_preds = run_forecast_loop(sales_history[:28], start_d=(current_d - 27))
        # Forecast starts tomorrow
        f_preds = run_forecast_loop(sales_history[28:], start_d=(current_d + 1))
        
        actuals = sales_history[28:]
        
        return {
            "item_id": str(item_id),
            "history": [float(x) for x in actuals],
            "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
            "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
            "metrics": {"accuracy": round(float(np.mean(actuals)), 2)}
        }
    except Exception as e:
        print(f"Prediction Crash: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {e}")

@app.get("/leaderboard")
def get_leaderboard_data():
    if os.path.exists(LEADERBOARD_PATH):
        return pd.read_csv(LEADERBOARD_PATH).head(100).fillna("N/A").to_dict(orient="records")
    return []

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)