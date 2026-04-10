import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from statsmodels.tsa.stattools import acf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
MODELS = {}
QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]

@app.on_event("startup")
def load_assets():
    print("🚀 Loading Models...")
    for q in QUANTILES:
        path = f"models/lgbm_q_{q}.pkl"
        if os.path.exists(path):
            MODELS[q] = joblib.load(path)

# --- Helper Logic ---

def calculate_stocking_risk(total_demand, current_inventory):
    """Business logic for inventory alerts."""
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
    df = pd.read_csv("data/processed_sales.csv")
    row = df[df['item_id'] == item_id]
    if row.empty: return [0.0] * count
    cols = [c for c in row.columns if c.startswith('d_')]
    return [float(x) for x in row[cols[-count:]].values.flatten()]

# --- Endpoints ---

@app.get("/leaderboard")
def get_leaderboard():
    LEADERBOARD_PATH = "data/item_leaderboard.csv"
    if os.path.exists(LEADERBOARD_PATH):
        df = pd.read_csv(LEADERBOARD_PATH)
        df['product_name'] = df['product_name'].fillna(df['item_id'])
        return df.to_dict(orient="records")
    raise HTTPException(status_code=404, detail="Leaderboard file missing.")

@app.get("/predict/{item_id}")
def predict(item_id: str, current_stock: int):
    full_h = get_history(item_id, 56)
    
    # 1. Backtest Loop (Last 28 days)
    bt_seed = np.array(full_h[:28]).copy()
    bt_preds = {str(q): [] for q in QUANTILES}
    for _ in range(28):
        x = bt_seed[-28:].reshape(1, -1)
        for q in QUANTILES:
            p = max(0.0, float(MODELS[q].predict(x)[0]))
            bt_preds[str(q)].append(p)
        bt_seed = np.append(bt_seed, bt_preds['0.5'][-1])

    # 2. Future Loop (Next 28 days)
    f_seed = np.array(full_h[-28:]).copy()
    f_preds = {str(q): [] for q in QUANTILES}
    for _ in range(28):
        x = f_seed[-28:].reshape(1, -1)
        for q in QUANTILES:
            p = max(0.0, float(MODELS[q].predict(x)[0]))
            f_preds[str(q)].append(p)
        f_seed = np.append(f_seed, f_preds['0.5'][-1])

    # 3. Stats & Risk Calculation
    actuals = np.array(full_h[-28:])
    
    # WSPL
    losses = []
    for q in QUANTILES:
        err = actuals - np.array(bt_preds[str(q)])
        losses.append(np.mean(np.where(err >= 0, q * err, (q-1) * err)))
    
    # Metrics
    cv = np.std(actuals) / np.mean(actuals) if np.mean(actuals) > 0 else 0
    try: lag_acf = acf(actuals, nlags=7, fft=True); seasonality = lag_acf[7]
    except: seasonality = 0
    within = sum(1 for i in range(28) if bt_preds['0.025'][i] <= actuals[i] <= bt_preds['0.975'][i])

    # --- GET PRODUCT NAME FOR FRONTEND ---
    df_map = pd.read_csv("data/item_leaderboard.csv")
    name_row = df_map[df_map['item_id'] == item_id]
    product_name = name_row['product_name'].iloc[0] if not name_row.empty else item_id

    # --- RISK ASSESSMENT ---
    total_expected_demand = sum(f_preds['0.5'])
    risk_info = calculate_stocking_risk(total_expected_demand, current_stock)

    return {
        "item_id": item_id,
        "product_name": product_name,
        "history": list(actuals),
        "backtest": bt_preds,
        "forecast": f_preds,
        "metrics": {
            "wspl": round(float(np.mean(losses)), 4),
            "backtest_accuracy": round((within/28)*100, 1),
            "mpiw": round(float(np.mean(np.array(f_preds['0.975']) - np.array(f_preds['0.025']))), 2),
            "cv": round(float(cv), 3),
            "seasonality_7d": round(float(seasonality), 3),
            "zero_pct": round(float((actuals == 0).sum()/28*100), 1)
        },
        "risk_assessment": risk_info
    }

if __name__ == "__main__":
    # Get the port from Railway's environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)