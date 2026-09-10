# frontend/components/presets.py
import numpy as np
import pandas as pd

FALLBACK_PRESETS = {
    "High Velocity": {
        "item_id": "FOODS_3_090",
        "blurb": "Fast-moving SKU with almost no zero-sale days.",
    },
    "Intermittent": {
        "item_id": "FOODS_1_043",
        "blurb": "Most days are zero, with occasional demand spikes.",
    },
    "Volatile": {
        "item_id": "FOODS_3_541",
        "blurb": "Large swings in daily volume — wide safety-stock bands.",
    },
    "Smooth": {
        "item_id": "FOODS_3_555",
        "blurb": "Stable everyday demand and tight prediction intervals.",
    },
}

PRESET_ORDER = ["High Velocity", "Intermittent", "Volatile", "Smooth"]


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def resolve_presets(df_items: pd.DataFrame | None) -> dict:
    """Pick one representative SKU per demand pattern from the leaderboard."""
    presets = {name: dict(meta) for name, meta in FALLBACK_PRESETS.items()}
    if df_items is None or df_items.empty or "item_id" not in df_items.columns:
        return presets

    items = df_items.drop_duplicates(subset=["item_id"]).copy()
    items["Popularity"] = _numeric(items, "Popularity")
    items["Volatility"] = _numeric(items, "Volatility")
    items["Sparsity"] = _numeric(items, "Sparsity")
    valid = items.dropna(subset=["Popularity"])
    if valid.empty:
        return presets

    known_ids = set(valid["item_id"])

    velocity = valid.sort_values("Popularity", ascending=False).iloc[0]
    presets["High Velocity"]["item_id"] = velocity["item_id"]

    vol_floor = valid["Popularity"].quantile(0.7)
    vol_pool = valid[valid["Popularity"] >= vol_floor]
    if vol_pool["Volatility"].notna().any():
        volatile = vol_pool.sort_values(
            ["Volatility", "Popularity"], ascending=[False, False]
        ).iloc[0]
        presets["Volatile"]["item_id"] = volatile["item_id"]

    inter_pool = valid[
        (valid["Sparsity"] >= 50)
        & (valid["Sparsity"] <= 90)
        & (valid["Popularity"] >= valid["Popularity"].quantile(0.4))
        & (valid["Volatility"].fillna(0) < 2.0)
        & (valid["item_id"] != presets["Volatile"]["item_id"])
    ].copy()
    if not inter_pool.empty:
        inter_pool["score"] = inter_pool["Sparsity"] * np.log1p(inter_pool["Popularity"])
        intermittent = inter_pool.sort_values("score", ascending=False).iloc[0]
        presets["Intermittent"]["item_id"] = intermittent["item_id"]

    vel_name = str(velocity.get("product_name", ""))
    smooth_pool = valid[
        (valid["Sparsity"].fillna(100) < 5)
        & (valid["Volatility"].fillna(99) < 0.4)
        & (valid["item_id"] != presets["High Velocity"]["item_id"])
    ]
    if not smooth_pool.empty:
        named = smooth_pool[smooth_pool["product_name"].astype(str) != vel_name]
        pick_from = named if not named.empty else smooth_pool
        smooth = pick_from.sort_values("Popularity", ascending=False).iloc[0]
        presets["Smooth"]["item_id"] = smooth["item_id"]

    for meta in presets.values():
        if meta["item_id"] not in known_ids:
            meta["item_id"] = velocity["item_id"]

    return presets


def pattern_caption(row: dict | None) -> str:
    if not row:
        return ""
    parts = []
    pop = pd.to_numeric(row.get("Popularity"), errors="coerce")
    vol = pd.to_numeric(row.get("Volatility"), errors="coerce")
    sparse = pd.to_numeric(row.get("Sparsity"), errors="coerce")
    if pd.notna(pop):
        parts.append(f"Volume {pop:,.0f}")
    if pd.notna(vol):
        parts.append(f"CV {vol:.2f}")
    if pd.notna(sparse):
        parts.append(f"{sparse:.0f}% zero days")
    return " · ".join(parts)
