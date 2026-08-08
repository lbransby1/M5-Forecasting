# backend/core/inference.py
import json
import os
import numpy as np
import lightgbm as lgb
from core.feature_engineering import compute_target_wday
from backend.core.data import (
    get_item_context,
    get_item_context_at_d,
    get_history,
    CALENDAR,
    MAPPINGS,
    PRODUCT_NAMES,
    MODEL_ARCH,
    HORIZON_MODEL_VERSION,
)

QUANTILES = ["0.005", "0.025", "0.165", "0.25", "0.5", "0.75", "0.835", "0.975", "0.995"]
MODELS = {}
_HORIZON_UNCERTAINTY_SCALES: list[float] | None = None

RECURSIVE_MODEL_DIR = "models/model_alpha"
HORIZON_MODEL_DIR_V1 = "models/model_horizon"
HORIZON_MODEL_DIR_V2 = "models/model_horizon_v2"


def _horizon_model_dir() -> str:
    if HORIZON_MODEL_VERSION == "v2":
        return HORIZON_MODEL_DIR_V2
    return HORIZON_MODEL_DIR_V1


def _model_path(quantile: str) -> str:
    if MODEL_ARCH == "horizon":
        return f"{_horizon_model_dir()}/global_model_alpha_{quantile}.txt"
    return f"{RECURSIVE_MODEL_DIR}/model_alpha_{quantile}.txt"


def load_ml_models():
    global MODELS
    MODELS = {}
    model_dir = _horizon_model_dir() if MODEL_ARCH == "horizon" else RECURSIVE_MODEL_DIR
    print(f"Loading {MODEL_ARCH} LightGBM models from {model_dir}...")
    for q in QUANTILES:
        path = _model_path(q)
        if os.path.exists(path):
            MODELS[q] = lgb.Booster(model_file=path)
    if len(MODELS) < len(QUANTILES):
        missing = [q for q in QUANTILES if q not in MODELS]
        raise RuntimeError(
            f"Expected {len(QUANTILES)} {MODEL_ARCH} models, loaded {len(MODELS)}. Missing: {missing}"
        )
    print(f"Loaded {len(MODELS)} quantile models ({MODEL_ARCH}, {HORIZON_MODEL_VERSION}).")


def _horizon_uncertainty_scales() -> list[float]:
    """Widen quantile bands with forecast distance (calibrated on validation MAE)."""
    global _HORIZON_UNCERTAINTY_SCALES
    if _HORIZON_UNCERTAINTY_SCALES is not None:
        return _HORIZON_UNCERTAINTY_SCALES

    scales_path = os.path.join(_horizon_model_dir(), "horizon_uncertainty_scales.json")
    if os.path.exists(scales_path):
        with open(scales_path, encoding="utf-8") as f:
            loaded = json.load(f)
        _HORIZON_UNCERTAINTY_SCALES = [float(loaded[str(h)]) for h in range(1, 29)]
    else:
        # Fallback: ~84% wider intervals by day 28 if calibration file is missing.
        _HORIZON_UNCERTAINTY_SCALES = [1.0 + 0.84 * np.sqrt((h - 1) / 27) for h in range(1, 29)]
    return _HORIZON_UNCERTAINTY_SCALES


def _apply_horizon_uncertainty_scaling(day_preds: dict[str, float], horizon_day: int) -> dict[str, float]:
    if MODEL_ARCH != "horizon" or HORIZON_MODEL_VERSION != "v2":
        return day_preds

    scale = _horizon_uncertainty_scales()[horizon_day - 1]
    if scale == 1.0:
        return day_preds

    median = day_preds["0.5"]
    return {q: max(0.0, median + (day_preds[q] - median) * scale) for q in QUANTILES}


def _encode_category(col: str, value) -> int:
    if not MAPPINGS.get(col):
        return 0
    encoded = MAPPINGS[col].get(str(value).strip())
    if encoded is None:
        return 0
    return int(encoded)


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


def _build_horizon_v1_feat_row(anchor_ctx, horizon_day: int, item_id: str, store_id: str):
    return [
        _encode_category("item_id", item_id),
        _encode_category("dept_id", anchor_ctx.get("dept_id")),
        _encode_category("cat_id", anchor_ctx.get("cat_id")),
        _encode_category("store_id", store_id),
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


def _build_horizon_v2_feat_row(anchor_ctx, horizon_day: int, item_id: str, store_id: str):
    target_wday = compute_target_wday(int(anchor_ctx.get("wday", 0)), horizon_day)
    return [
        _encode_category("item_id", item_id),
        _encode_category("dept_id", anchor_ctx.get("dept_id")),
        _encode_category("cat_id", anchor_ctx.get("cat_id")),
        _encode_category("store_id", store_id),
        _encode_category("state_id", anchor_ctx.get("state_id")),
        int(anchor_ctx.get("wday", 0)),
        int(anchor_ctx.get("month", 0)),
        float(anchor_ctx.get("sell_price", 0)),
        float(anchor_ctx.get("price_norm", 0)),
        int(anchor_ctx.get("snap_CA", 0)),
        int(anchor_ctx.get("snap_TX", 0)),
        int(anchor_ctx.get("snap_WI", 0)),
        float(anchor_ctx.get("price_momentum_7d", 0)),
        float(anchor_ctx.get("price_momentum_28d", 0)),
        float(anchor_ctx.get("lag_28", 0)),
        float(anchor_ctx.get("roll_mean_7_lag_28", 0)),
        float(anchor_ctx.get("roll_mean_28_lag_28", 0)),
        float(anchor_ctx.get("masked_roll_mean_28_lag_28", 0)),
        float(anchor_ctx.get("ema_lag_28", 0)),
        float(anchor_ctx.get("days_since_last_sale_lag_28", 0)),
        horizon_day,
        _encode_category("target_wday", target_wday),
    ]


def _build_horizon_feat_row(anchor_ctx, horizon_day: int, item_id: str, store_id: str):
    if HORIZON_MODEL_VERSION == "v2":
        return _build_horizon_v2_feat_row(anchor_ctx, horizon_day, item_id, store_id)
    return _build_horizon_v1_feat_row(anchor_ctx, horizon_day, item_id, store_id)


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


def run_horizon_forecast(anchor_ctx, item_id: str, store_id: str):
    preds = {q: [] for q in QUANTILES}
    for h in range(1, 29):
        day_preds = _predict_quantiles(_build_horizon_feat_row(anchor_ctx, h, item_id, store_id))
        day_preds = _apply_horizon_uncertainty_scaling(day_preds, h)
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


def _validate_context(ctx: dict, item_id: str, store_id: str, sales_history):
    ctx_item = str(ctx.get("item_id", ""))
    ctx_store = str(ctx.get("store_id", ""))
    if ctx_item and ctx_item != str(item_id):
        print(f"[warn] Redis ctx item_id mismatch: ctx={ctx_item} request={item_id}")
    if ctx_store and ctx_store != str(store_id):
        print(f"[warn] Redis ctx store_id mismatch: ctx={ctx_store} request={store_id}")

    lag_28 = float(ctx.get("lag_28", 0))
    recent_sales = float(np.sum(sales_history[-28:]))
    if lag_28 == 0.0 and recent_sales > 0:
        raise ValueError(
            f"Missing lag_28 features for {item_id}/{store_id}. "
            "Reload Redis: python training/load_feature_store.py --force"
        )


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

    bt_preds = run_horizon_forecast(anchor_ctx, item_id, store_id)
    f_preds = run_horizon_forecast(ctx, item_id, store_id)
    return bt_preds, f_preds


def generate_full_forecast(item_id: str, store_id: str):
    if not MODELS:
        raise ValueError("Models not loaded.")

    ctx = get_item_context(item_id, store_id)
    if not ctx:
        raise ValueError("Item/Store combination not found.")

    sales_history = _prepare_history(item_id, store_id)
    _validate_context(ctx, item_id, store_id, sales_history)

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
