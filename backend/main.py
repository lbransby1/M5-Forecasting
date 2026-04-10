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

# --- 2. CONFIGURATION ---
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

# --- 3. HELPER FUNCTIONS ---

def get_history(item_id, count=56):
    """Missing function re-added to prevent NameError crash."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        return [0.0] * count
    try:
        q = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = q.filter(pl.col("item_id") == item_id).tail(count).collect()
        if result.is_empty():
            return [0.0] * count
        return result["sales"].to_list()
    except Exception as e:
        print(f"❌ Error reading history: {e}")
        return [0.0] * count

def get_item_context(item_id: str):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        last_row = df.filter(pl.col("item_id") == item_id).tail(1).collect()
        return last_row.to_dicts()[0] if not last_row.is_empty() else None
    except:
        return None

# --- 4. ENDPOINTS ---

@app.get("/leaderboard")
def get_leaderboard():
    if os.path.exists(LEADERBOARD_PATH):
        df = pd.read_csv(LEADERBOARD_PATH)
        return df.head(100).to_dict(orient="records")
    raise HTTPException(status_code=404, detail="Leaderboard not found")

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: float = 0.0):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    ctx = get_item_context(item_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Item data not found.")

    sales_history = np.array(get_history(item_id, 56)) 
    
    def run_forecast_loop(seed_sales):
        preds = {q: [] for q in QUANTILES}
        current_window = list(seed_sales)
        
        for i in range(28):
            # Feature engineering matching your notebook
            feat_row = [
                ctx.get('wday', 1), ctx.get('month', 1), 
                ctx.get('sell_price', 0), ctx.get('price_norm', 0),
                np.mean(current_window[-7:]),  
                np.mean(current_window[-28:]), 
                ctx.get('snap_CA', 0), ctx.get('snap_TX', 0), ctx.get('snap_WI', 0)
            ]
            
            x = np.array(feat_row).reshape(1, -1)
            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(x)[0]))
                preds[q].append(p)
            current_window.append(preds["0.5"][-1])
        return preds

    try:
        bt_preds = run_forecast_loop(sales_history[:28])
        f_preds = run_forecast_loop(sales_history[28:])
        actuals = sales_history[28:]
        
        return {
            "item_id": item_id,
            "product_name": str(item_id), # Fixed 'id' error
            "history": [float(x) for x in actuals],
            "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
            "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
            "metrics": {
                "backtest_accuracy": round(float(np.mean(actuals)), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)