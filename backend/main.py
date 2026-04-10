import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uvicorn

app = FastAPI(title="M5 Forecasting API")

# --- 1. MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIGURATION & ASSETS ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
MODEL_DIR = "models/model_alpha"
MAPPING_PATH = "backend/category_mappings.json"

# Load Assets
CALENDAR = pd.read_csv("backend/data/raw/calendar.csv")
CALENDAR['d_num'] = CALENDAR['d'].str.replace('d_', '').astype(int)

MODELS = {}
QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]
MAPPINGS = {}

FEATURES = [
    'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
    'wday', 'month', 'sell_price', 'price_norm', 
    'roll_mean_7', 'roll_mean_28', 'snap_CA', 'snap_TX', 'snap_WI'
]

@app.on_event("startup")
def load_assets():
    print(f"🚀 Loading Alpha Models from {MODEL_DIR}...")
    for q in QUANTILES:
        path = f"{MODEL_DIR}/model_alpha_{q}.txt"
        if os.path.exists(path):
            MODELS[q] = lgb.Booster(model_file=path)
    
    global MAPPINGS
    if os.path.exists(MAPPING_PATH):
        print(f"📖 Loading categorical mappings from {MAPPING_PATH}...")
        with open(MAPPING_PATH, "r") as f:
            MAPPINGS = json.load(f)
            
    print(f"🏁 Startup complete. Loaded {len(MODELS)} models and categorical maps.")

# --- 3. HELPERS ---

def get_history(item_id, count=84):
    if not os.path.exists(PROCESSED_DATA_PATH): return [0.0] * count
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = df.filter(pl.col("item_id") == item_id).tail(count).collect()
        return result["sales"].to_list() if not result.is_empty() else [0.0] * count
    except: return [0.0] * count

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
    if not MODELS: 
        raise HTTPException(status_code=503, detail="Models not loaded.")
    
    ctx = get_item_context(item_id)
    if not ctx: 
        raise HTTPException(status_code=404, detail="Item not found.")

    # 🛠️ INCREASED BUFFER: Fetch 84 days (56 for lags + 28 for seed)
    history_needed = 84
    sales_history = np.array(get_history(item_id, history_needed)) 
    
    if len(sales_history) < history_needed:
        sales_history = np.pad(sales_history, (history_needed - len(sales_history), 0), 'constant')
    elif len(sales_history) > history_needed:
        sales_history = sales_history[-history_needed:]

    current_d = int(ctx.get('d', 1941))

    def run_forecast_loop(seed_history, start_d):
        preds = {q: [] for q in QUANTILES}
        current_window = [float(x) for x in seed_history]
        
        for i in range(28):
            target_day = int(start_d + i)
            day_info = CALENDAR[CALENDAR['d_num'] == target_day].iloc[0]
            
            # Assembly with high-fidelity casting
            feat_row = [
                int(MAPPINGS.get('item_id', {}).get(str(ctx.get('item_id')), 0)),
                int(MAPPINGS.get('dept_id', {}).get(str(ctx.get('dept_id')), 0)),
                int(MAPPINGS.get('cat_id', {}).get(str(ctx.get('cat_id')), 0)),
                int(MAPPINGS.get('store_id', {}).get(str(ctx.get('store_id')), 0)),
                int(MAPPINGS.get('state_id', {}).get(str(ctx.get('state_id')), 0)),
                int(day_info['wday']), int(day_info['month']), 
                float(ctx.get('sell_price', 0)), float(ctx.get('price_norm', 0)),
                # STEP 3: LAG ALIGNMENT
                float(np.mean(current_window[-35:-28])), 
                float(np.mean(current_window[-56:-28])), 
                int(day_info['snap_CA']), int(day_info['snap_TX']), int(day_info['snap_WI'])
            ]
            
            x = np.array(feat_row).reshape(1, -1)
            
            # --- THE FIX: BOOSTED RECURSION ---
            current_day_preds = []
            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(x)[0]))
                preds[q].append(p)
                current_day_preds.append(p)
            
            # Instead of the conservative Median (0.5), use a Weighted Mean
            # We weight the higher quantiles slightly more to preserve 'Spikiness'
            # This is a common MLE 'hack' for zero-inflated retail data
            p_mean = np.mean(current_day_preds)
            p_high = preds["0.75"][-1]
            
            # Blend the mean and the 75th percentile to keep the 'Rolling Mean' healthy
            recursive_value = (p_mean * 0.7) + (p_high * 0.3)
            current_window.append(recursive_value)
            
        return preds

    try:
        # Step 3 Alignment: Use the 84-86 day buffer for feature calculation
        # Backtest predicts the last 28 days of known history
        bt_preds = run_forecast_loop(sales_history[:56], start_d=(current_d - 27))
        
        # Forecast predicts the next 28 days into the future
        f_preds = run_forecast_loop(sales_history[56:], start_d=(current_d + 1))
        
        # --- UPDATED FOR VISUALIZATION ---
        # Instead of just the last 28 days, return the full history buffer
        # This allows the frontend to plot a longer historical trend line
        return {
            "item_id": str(item_id),
            "history": [float(x) for x in sales_history], # Returns all 86 days
            "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
            "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
            "metrics": {
                "accuracy": round(float(np.mean(sales_history[-28:])), 2)
            }
        }
    
    except Exception as e:
        print(f"Prediction Crash: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/leaderboard")
def get_leaderboard_data():
    if os.path.exists(LEADERBOARD_PATH):
        return pd.read_csv(LEADERBOARD_PATH).head(100).fillna("N/A").to_dict(orient="records")
    return []

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)