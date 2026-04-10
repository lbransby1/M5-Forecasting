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
        # Optimization: collect only what we need
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = df.filter(pl.col("item_id") == item_id).tail(count).collect()
        if result.is_empty():
            return [0.0] * count
        return result["sales"].to_list()
    except:
        return [0.0] * count

def get_item_context(item_id: str):
    if not os.path.exists(PROCESSED_DATA_PATH): return None
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        last_row = df.filter(pl.col("item_id") == item_id).tail(1).collect()
        if last_row.is_empty(): return None
        
        # Convert to dict and handle categorical/string types
        ctx = last_row.to_dicts()[0]
        
        # CRITICAL: If your model expects numbers for IDs, 
        # you need to extract the underlying category index.
        # This code assumes Polars/LightGBM handled the string-to-cat conversion.
        return ctx
    except: return None

# --- PREDICTION ---

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: float = 0.0):
    if not MODELS: raise HTTPException(status_code=503, detail="Models not loaded.")
    
    ctx = get_item_context(item_id)
    if not ctx: raise HTTPException(status_code=404, detail="Item not found.")

    # Get 56 days: 28 for backtest seed, 28 for current history
    sales_history = np.array(get_history(item_id, 56)) 
    
    if len(sales_history) < 56:
        # Pad with zeros if history is short to prevent slice errors
        sales_history = np.pad(sales_history, (56 - len(sales_history), 0), 'constant')

    def run_forecast_loop(seed_sales):
        preds = {q: [] for q in QUANTILES}
        current_window = list(seed_sales)
        
        for i in range(28):
            # 14 FEATURES - EXACT ORDER MATTERS
            feat_row = [
                ctx.get('item_id'), ctx.get('dept_id'), ctx.get('cat_id'), 
                ctx.get('store_id'), ctx.get('state_id'),
                ctx.get('wday', 1), ctx.get('month', 1), 
                ctx.get('sell_price', 0), ctx.get('price_norm', 0),
                np.mean(current_window[-7:]),  # roll_mean_7
                np.mean(current_window[-28:]), # roll_mean_28
                # ADD THE MISSING 3 FEATURES TO REACH 14
                current_window[-1],  # lag_1 (Immediate momentum)
                current_window[-7],  # lag_7 (Weekly seasonality)
                ctx.get('snap_CA', 0) if ctx.get('state_id') == 'CA' else (ctx.get('snap_TX') if ctx.get('state_id') == 'TX' else ctx.get('snap_WI', 0))
            ]
            
            # Use the DataFrame fix to handle categoricals properly
            df_exec = pd.DataFrame([feat_row], columns=[
                'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                'wday', 'month', 'sell_price', 'price_norm', 
                'roll_mean_7', 'roll_mean_28', 'lag_1', 'lag_7', 'snap'
            ])
            
            # Cast categoricals
            for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
                df_exec[col] = df_exec[col].astype('category')

            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(df_exec)[0]))
                preds[q].append(p)
            
            current_window.append(preds["0.5"][-1])
            
        return preds

    try:
        # Correct Slicing: 
        # bt_seed = first 28 days (0-27)
        # f_seed = last 28 days (28-55)
        bt_preds = run_forecast_loop(sales_history[:28])
        f_preds = run_forecast_loop(sales_history[28:])
        actuals = sales_history[28:]
        
        return {
            "item_id": str(item_id),
            "history": [float(x) for x in actuals],
            "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
            "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
            "metrics": {"accuracy": round(float(np.mean(actuals)), 2)}
        }
    except Exception as e:
        # This will show you exactly which line failed in the Railway logs
        print(f"Prediction Crash: {e}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {e}")

@app.get("/leaderboard")
def get_leaderboard_data():
    if os.path.exists(LEADERBOARD_PATH):
        return pd.read_csv(LEADERBOARD_PATH).head(100).fillna("N/A").to_dict(orient="records")
    return []

if __name__ == "__main__":
    # Use Railway's preferred port
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)