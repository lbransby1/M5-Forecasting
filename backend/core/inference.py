# backend/core/inference.py
import os
import numpy as np
import lightgbm as lgb
from backend.core.data import (
    get_item_context,
    get_item_context_at_d,
    get_history,
    CALENDAR,
    MAPPINGS,
    PRODUCT_NAMES,
    MODEL_ARCH,
)

QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]
MODELS = {}

RECURSIVE_MODEL_DIR = "models/model_alpha"
HORIZON_MODEL_DIR = "models/model_horizon"


def _model_path(quantile: str) -> str:
    if MODEL_ARCH == "horizon":
        return f"{HORIZON_MODEL_DIR}/global_model_alpha_{quantile}.txt"
    return f"{RECURSIVE_MODEL_DIR}/model_alpha_{quantile}.txt"


def load_ml_models():
    global MODELS
    MODELS = {}
    model_dir = HORIZON_MODEL_DIR if MODEL_ARCH == "horizon" else RECURSIVE_MODEL_DIR
    print(f"🚀 Loading {MODEL_ARCH} LightGBM models from {model_dir}...")
    for q in QUANTILES:
        path = _model_path(q)
        if os.path.exists(path):
            MODELS[q] = lgb.Booster(model_file=path)
    if len(MODELS) < len(QUANTILES):
        missing = [q for q in QUANTILES if q not in MODELS]
        raise RuntimeError(
            f"Expected {len(QUANTILES)} {MODEL_ARCH} models, loaded {len(MODELS)}. Missing: {missing}"
        )
    print(f"✅ Loaded {len(MODELS)} quantile models ({MODEL_ARCH} architecture).")


def _encode_category(col: str, value) -> int:
    return int(MAPPINGS.get(col, {}).get(str(value), 0))


def _build_recursive_feat_row(ctx, day_info, current_window):
    return [
        _encode_category("item_id", ctx.get("item_id")),
        _encode_category("dept_id", ctx.get("dept_id")),
        _encode_category("cat_id", ctx.get("cat_id")),
        _encode_category("store_id", ctx.get("store_id")),
        _encode_category("state_id", ctx.get("state_id")),
        int(day_info["wday"]),
        int(day_info["month"]),
        float(ctx.get("sell_price", 0)),
        float(ctx.get("price_norm", 0)),
        float(np.mean(current_window[-35:-28])),
        float(np.mean(current_window[-56:-28])),
        int(day_info["snap_CA"]),
        int(day_info["snap_TX"]),
        int(day_info["snap_WI"]),
    ]


def _build_horizon_feat_row(anchor_ctx, horizon_day: int):
    return [
        _encode_category("item_id", anchor_ctx.get("item_id")),
        _encode_category("dept_id", anchor_ctx.get("dept_id")),
        _encode_category("cat_id", anchor_ctx.get("cat_id")),
        _encode_category("store_id", anchor_ctx.get("store_id")),
        _encode_category("state_id", anchor_ctx.get("state_id")),
        int(anchor_ctx.get("wday", 0)),
        int(anchor_ctx.get("month", 0)),
        float(anchor_ctx.get("sell_price", 0)),
        float(anchor_ctx.get("price_norm", 0)),
        float(anchor_ctx.get("roll_mean_7", 0)),
        float(anchor_ctx.get("roll_mean_28", 0)),
        int(anchor_ctx.get("snap_CA", 0)),
        int(anchor_ctx.get("snap_TX", 0)),
        int(anchor_ctx.get("snap_WI", 0)),
        horizon_day,
    ]


def _predict_quantiles(feat_row):
    x = np.array(feat_row).reshape(1, -1)
    return {q: max(0.0, float(MODELS[q].predict(x)[0])) for q in QUANTILES}


def run_forecast_loop(seed_history, start_d, ctx):
    preds = {q: [] for q in QUANTILES}
    current_window = [float(x) for x in seed_history]

    for i in range(28):
        target_day = int(start_d + i)
        day_matches = CALENDAR[CALENDAR["d_num"] == target_day]
        day_info = day_matches.iloc[0] if not day_matches.empty else CALENDAR.iloc[-1]

        day_preds = _predict_quantiles(_build_recursive_feat_row(ctx, day_info, current_window))
        for q in QUANTILES:
            preds[q].append(day_preds[q])

        p_mean = np.mean(list(day_preds.values()))
        p_high = day_preds["0.75"]
        current_window.append((p_mean * 0.7) + (p_high * 0.3))

    return preds


def run_horizon_forecast(anchor_ctx):
    preds = {q: [] for q in QUANTILES}
    for h in range(1, 29):
        day_preds = _predict_quantiles(_build_horizon_feat_row(anchor_ctx, h))
        for q in QUANTILES:
            preds[q].append(day_preds[q])
    return preds


def _prepare_history(item_id: str, store_id: str):
    history_needed = 84
    sales_history = np.array(get_history(item_id, store_id, history_needed))

    if len(sales_history) < history_needed:
        sales_history = np.pad(sales_history, (history_needed - len(sales_history), 0), "constant")
    elif len(sales_history) > history_needed:
        sales_history = sales_history[-history_needed:]

    return sales_history


def _generate_recursive_forecast(item_id: str, store_id: str, ctx, sales_history):
    current_d = int(ctx.get("d", 1941))
    bt_preds = run_forecast_loop(sales_history[:56], start_d=(current_d - 27), ctx=ctx)
    f_preds = run_forecast_loop(sales_history[56:], start_d=(current_d + 1), ctx=ctx)
    return bt_preds, f_preds


def _generate_horizon_forecast(item_id: str, store_id: str, ctx, sales_history):
    current_d = int(ctx.get("d", 1941))
    anchor_d = current_d - 28

    anchor_ctx = get_item_context_at_d(item_id, store_id, anchor_d)
    if not anchor_ctx:
        raise ValueError(f"No anchor context found at d={anchor_d} for backtest.")

    bt_preds = run_horizon_forecast(anchor_ctx)
    f_preds = run_horizon_forecast(ctx)
    return bt_preds, f_preds


def generate_full_forecast(item_id: str, store_id: str):
    if not MODELS:
        raise ValueError("Models not loaded.")

    ctx = get_item_context(item_id, store_id)
    if not ctx:
        raise ValueError("Item/Store combination not found.")

    sales_history = _prepare_history(item_id, store_id)

    if MODEL_ARCH == "horizon":
        bt_preds, f_preds = _generate_horizon_forecast(item_id, store_id, ctx, sales_history)
    else:
        bt_preds, f_preds = _generate_recursive_forecast(item_id, store_id, ctx, sales_history)

    return {
        "item_id": str(item_id),
        "store_id": str(store_id),
        "product_name": PRODUCT_NAMES.get(item_id, item_id),
        "history": [float(x) for x in sales_history],
        "backtest": {k: [float(x) for x in v] for k, v in bt_preds.items()},
        "forecast": {k: [float(x) for x in v] for k, v in f_preds.items()},
        "metrics": {"accuracy": round(float(np.mean(sales_history[-28:])), 2)},
    }
