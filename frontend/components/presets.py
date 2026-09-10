# frontend/components/presets.py
import pandas as pd

# IDs verified against the production /predict endpoint (CA_3).
# Leaderboard stats alone are not enough — some high-vol / sparse SKUs are not in Redis.
FALLBACK_PRESETS = {
    "High Velocity": {
        "item_id": "FOODS_3_090",
        "blurb": "Fast-moving SKU with almost no zero-sale days.",
    },
    "Intermittent": {
        "item_id": "FOODS_3_785",
        "blurb": "Most days are zero, with occasional demand spikes.",
    },
    "Volatile": {
        "item_id": "FOODS_3_547",
        "blurb": "Uneven daily volume — wider safety-stock bands.",
    },
    "Smooth": {
        "item_id": "FOODS_3_555",
        "blurb": "Stable everyday demand and tight prediction intervals.",
    },
}

PRESET_ORDER = ["High Velocity", "Intermittent", "Volatile", "Smooth"]


def resolve_presets(df_items: pd.DataFrame | None) -> dict:
    """Return demo SKUs that exist in the feature store."""
    return {name: dict(meta) for name, meta in FALLBACK_PRESETS.items()}


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
