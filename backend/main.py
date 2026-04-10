import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI(title="M5 Forecasting API")

# --- 1. MIDDLEWARE (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIGURATION & PATHS ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
MODEL_DIR = "models/model_alpha"

MODELS = {}
# Using string keys to keep JSON serialization consistent
QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]

@app.on_event("startup")
def load_assets():
    print(f"🚀 Loading Alpha Models from {MODEL_DIR}...")
    models_loaded = 0
    
    for q in QUANTILES:
        path = f"{MODEL_DIR}/model_alpha_{q}.txt"
        if os.path.exists(path):
            try:
                MODELS[q] = lgb.Booster(model_file=path)
                models_loaded += 1
                print(f"✅ Loaded: {path}")
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
        else:
            print(f"⚠️ Missing model: {path}")
            
    print(f"🏁 Startup complete. Loaded {models_loaded}/{len(QUANTILES)} models.")

# --- 3. HELPER LOGIC ---

def calculate_stocking_risk(total_demand, current_inventory):
    total_demand = float(total_demand)
    current_inv = float(current_inventory)
    
    if current_inv < (total_demand * 0.5):
        return {"level": "CRITICAL", "action": f"Restock {int(total_demand - current_inv) + 10} units immediately."}
    elif current_inv < total_demand:
        return {"level": "Warning", "action": "Stock is low - replenishment recommended."}
    elif current_inv > (total_demand * 2.5):
        return {"level": "Overstock", "action": "Excess inventory - pause orders."}
    
    return {"level": "Healthy", "action": "Inventory levels aligned with 28-day demand."}

def get_history(item_id, count=56):
    """Optimized data reading using Polars Lazy-loading."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        return [0.0] * count
    
    try:
        # Use scan_parquet + collect to save significant RAM (prevents OOM)
        q = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = (
            q.filter(pl.col("item_id") == item_id)
            .tail(count)
            .collect()
        )
        if result.is_empty():
            return [0.0] * count
        return result["sales"].to_list()
    except Exception as e:
        print(f"❌ Error reading history: {e}")
        return [0.0] * count

# --- 4. ENDPOINTS ---

@app.get("/")
def root():
    import datetime
    data_time = "Unknown"
    if os.path.exists(PROCESSED_DATA_PATH):
        mtime = os.path.getmtime(PROCESSED_DATA_PATH)
        data_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "status": "online",
        "models_loaded": len(MODELS.keys()),
        "data_updated": data_time
    }

@app.get("/leaderboard")
def get_leaderboard():
    if os.path.exists(LEADERBOARD_PATH):
        try:
            df = pd.read_csv(LEADERBOARD_PATH)
            df['product_name'] = df['product_name'].fillna(df['item_id'])
            return df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")
    raise HTTPException(status_code=404, detail="Leaderboard missing")

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: float = 0.0):
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    # Get history: 28 for backtest seed + 28 for final history = 56
    full_h = get_history(item_id, 56)
    
    try:
        # --- 1. BACKTEST (Features=14) ---
        # Seed starts at t-28, we need 14 prior days for features
        bt_preds = {str(q): [] for q in QUANTILES}
        # Slice full_h to get the 14 days preceding the last 28 days
        seed = np.array(full_h[14:28]).copy() 
        
        for _ in range(28):
            x = seed[-14:].reshape(1, -1) # Feed 14 features
            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(x)[0]))
                bt_preds[q].append(p)
            # Append 0.5 quantile to the seed to forecast next step
            seed = np.append(seed, bt_preds["0.5"][-1])

        # --- 2. FUTURE FORECAST (Features=14) ---
        f_preds = {str(q): [] for q in QUANTILES}
        seed = np.array(full_h[-14:]).copy() # Use the latest 14 days
        
        for _ in range(28):
            x = seed[-14:].reshape(1, -1) # Feed 14 features
            for q in QUANTILES:
                p = max(0.0, float(MODELS[q].predict(x)[0]))
                f_preds[q].append(p)
            seed = np.append(seed, f_preds["0.5"][-1])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {e}")

    # --- 3. METRICS ---
    actuals = np.array(full_h[-28:])

    losses = []
    for q_str in QUANTILES:
        q = float(q_str)
        # Convert to float() to avoid numpy types
        err = actuals - np.array(bt_preds[q_str])
        losses.append(np.mean(np.where(err >= 0, q * err, (q-1) * err)))
    
    # Ensure these are standard Python ints/floats
    backtest_acc = float((sum(1 for i in range(28) if bt_preds['0.025'][i] <= actuals[i] <= bt_preds['0.975'][i]) / 28) * 100)
    avg_loss = float(np.mean(losses))
    mpiw_val = float(np.mean(np.array(f_preds['0.975']) - np.array(f_preds['0.025'])))

    total_expected_demand = float(sum(f_preds['0.5']))
    risk_info = calculate_stocking_risk(total_expected_demand, current_stock)

    return {
        "item_id": item_id,
        "product_name": str(id),
        "history": [float(x) for x in actuals], # Convert list to floats
        "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()}, # Convert dict lists
        "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()}, # Convert dict lists
        "metrics": {
            "wspl": round(avg_loss, 4),
            "backtest_accuracy": round(backtest_acc, 1),
            "mpiw": round(mpiw_val, 2)
        },
        "risk_assessment": risk_info
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)