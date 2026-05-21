# backend/core/inference.py
import os
import numpy as np
import lightgbm as lgb
from core.data import get_item_context, get_history, CALENDAR, MAPPINGS, PRODUCT_NAMES

MODEL_DIR = "models/model_alpha"
QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]
MODELS = {}

def load_ml_models():
    print(f"🚀 Loading LightGBM Models from {MODEL_DIR}...")
    for q in QUANTILES:
        path = f"{MODEL_DIR}/model_alpha_{q}.txt"
        if os.path.exists(path):
            MODELS[q] = lgb.Booster(model_file=path)
    print(f"✅ Loaded {len(MODELS)} quantile models.")

def run_forecast_loop(seed_history, start_d, ctx):
    preds = {q: [] for q in QUANTILES}
    current_window = [float(x) for x in seed_history]
    
    for i in range(28):
        target_day = int(start_d + i)
        day_matches = CALENDAR[CALENDAR['d_num'] == target_day]
        day_info = day_matches.iloc[0] if not day_matches.empty else CALENDAR.iloc[-1]
        
        feat_row = [
            int(MAPPINGS.get('item_id', {}).get(str(ctx.get('item_id')), 0)),
            int(MAPPINGS.get('dept_id', {}).get(str(ctx.get('dept_id')), 0)),
            int(MAPPINGS.get('cat_id', {}).get(str(ctx.get('cat_id')), 0)),
            int(MAPPINGS.get('store_id', {}).get(str(ctx.get('store_id')), 0)),
            int(MAPPINGS.get('state_id', {}).get(str(ctx.get('state_id')), 0)),
            int(day_info['wday']), int(day_info['month']), 
            float(ctx.get('sell_price', 0)), float(ctx.get('price_norm', 0)),
            float(np.mean(current_window[-35:-28])), 
            float(np.mean(current_window[-56:-28])), 
            int(day_info['snap_CA']), int(day_info['snap_TX']), int(day_info['snap_WI'])
        ]
        
        x = np.array(feat_row).reshape(1, -1)
        current_day_preds = []
        for q in QUANTILES:
            p = max(0.0, float(MODELS[q].predict(x)[0]))
            preds[q].append(p)
            current_day_preds.append(p)
        
        p_mean = np.mean(current_day_preds)
        p_high = preds["0.75"][-1]
        current_window.append((p_mean * 0.7) + (p_high * 0.3))
        
    return preds

def generate_full_forecast(item_id: str, store_id: str):
    if not MODELS: raise ValueError("Models not loaded.")
    
    ctx = get_item_context(item_id, store_id)
    if not ctx: raise ValueError("Item/Store combination not found.")

    history_needed = 84
    sales_history = np.array(get_history(item_id, store_id, history_needed)) 
    
    if len(sales_history) < history_needed:
        sales_history = np.pad(sales_history, (history_needed - len(sales_history), 0), 'constant')
    elif len(sales_history) > history_needed:
        sales_history = sales_history[-history_needed:]

    current_d = int(ctx.get('d', 1941))
    
    bt_preds = run_forecast_loop(sales_history[:56], start_d=(current_d - 27), ctx=ctx)
    f_preds = run_forecast_loop(sales_history[56:], start_d=(current_d + 1), ctx=ctx)
    
    return {
        "item_id": str(item_id),
        "store_id": str(store_id),
        "product_name": PRODUCT_NAMES.get(item_id, item_id),
        "history": [float(x) for x in sales_history],
        "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
        "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
        "metrics": {"accuracy": round(float(np.mean(sales_history[-28:])), 2)}
    }