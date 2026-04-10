import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(title="M5 Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
MODEL_DIR = "models/model_alpha"

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

# --- HELPERS ---

def get_history(item_id, count=56):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return [0.0] * count
    try:
        q = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = q.filter(pl.col("item_id") == item_id).tail(count).collect()
        return result["sales"].to_list() if not result.is_empty() else [0.0] * count
    except:
        return [0.0] * count

def get_item_context(item_id: str):
    """Fetches the latest state of an item (prices, snap, calendar context)."""
    if not os.path.exists(PROCESSED_DATA_PATH): return None
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        last_row = df.filter(pl.col("item_id") == item_id).tail(1).collect()
        return last_row.to_dicts()[0] if not last_row.is_empty() else None
    except: return None

# --- PREDICTION ---

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: float = 0.0):
    if not MODELS: raise HTTPException(status_code=503, detail="Models not loaded.")
    
    ctx = get_item_context(item_id)
    if not ctx: raise HTTPException(status_code=404, detail="Item not found.")

    sales_history = np.array(get_history(item_id, 56)) 
    
    def run_forecast_loop(seed_sales):
        preds = {q: [] for q in QUANTILES}
        current_window = list(seed_sales)
        
        for i in range(28):
            # BUILD THE EXACT 14-FEATURE VECTOR 
            # Order matches your list: item, dept, cat, store, state, calendar, rolling
            feat_row = [
                ctx.get('item_id'),   # 1
                ctx.get('dept_id'),   # 2
                ctx.get('cat_id'),    # 3
                ctx.get('store_id'),  # 4
                ctx.get('state_id'),  # 5
                ctx.get('wday', 1),    # 6
                ctx.get('month', 1),   # 7
                ctx.get('sell_price', 0), # 8
                ctx.get('price_norm', 0), # 9
                np.mean(current_window[-7:]),  # 10: roll_mean_7
                np.mean(current_window[-28:]), # 11: roll_mean_28
                ctx.get('snap_CA', 0), # 12
                ctx.get('snap_TX', 0), # 13
                ctx.get('snap_WI', 0)  # 14
            ]
            
            # Convert categorical strings/ints to the format LightGBM expects
            # If your model used LabelEncoding, ensure ctx values are integers.
            x = np.array(feat_row).reshape(1, -1)
            
            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(x)[0]))
                preds[q].append(p)
            
            # Update the sliding window with the median forecast
            current_window.append(preds["0.5"][-1])
            
        return preds
    
    
    try:
        bt_preds = run_forecast_loop(sales_history[:28])
        f_preds = run_forecast_loop(sales_history[28:])
        actuals = sales_history[28:]
        
        return {
            "item_id": item_id,
            "product_name": str(item_id),
            "history": [float(x) for x in actuals],
            "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
            "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
            "metrics": {"accuracy": round(float(np.mean(actuals)), 2)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {e}")

@app.get("/leaderboard")
def leaderboard():
    if os.path.exists(LEADERBOARD_PATH):
        return pd.read_csv(LEADERBOARD_PATH).head(100).to_dict(orient="records")
    return []

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))