# backend/core/data.py
import os
import json
import pandas as pd
import polars as pl

from backend.core import redis_store

# --- CONFIGURATION ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
PRODUCT_MAP_PATH = f"{DATA_DIR}/product_map.json"

MODEL_ARCH = os.environ.get("MODEL_ARCH", "recursive")
FEATURE_STORE = os.environ.get("FEATURE_STORE", "parquet")
HORIZON_MODEL_VERSION = os.environ.get("HORIZON_MODEL_VERSION", "v2")


def _default_mapping_path():
    if MODEL_ARCH == "horizon":
        if HORIZON_MODEL_VERSION == "v2":
            return "models/model_horizon_v2/category_mappings.json"
        return "models/model_horizon/category_mappings.json"
    return "backend/category_mappings.json"


MAPPING_PATH = os.environ.get("CATEGORY_MAPPINGS_PATH", _default_mapping_path())

# --- GLOBAL ASSETS ---
CALENDAR = pd.read_csv(f"{DATA_DIR}/raw/calendar.csv")
CALENDAR["d_num"] = CALENDAR["d"].str.replace("d_", "").astype(int)
MAPPINGS = {}
PRODUCT_NAMES = {}


def init_feature_store():
    if FEATURE_STORE == "redis":
        redis_store.init_redis()
        print(f"Redis feature store connected ({redis_store.REDIS_URL})")


def load_data_assets():
    global MAPPINGS, PRODUCT_NAMES
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as f:
            MAPPINGS = json.load(f)
        print(f"Category mappings loaded from {MAPPING_PATH}")
    else:
        print(f"Category mappings not found at {MAPPING_PATH}")
    if os.path.exists(PRODUCT_MAP_PATH):
        with open(PRODUCT_MAP_PATH, "r") as f:
            PRODUCT_NAMES = json.load(f)
    print(f"Data assets loaded: {len(PRODUCT_NAMES)} product names.")


def _parquet_history(item_id: str, store_id: str, count=84):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return [0.0] * count
    df = pl.scan_parquet(PROCESSED_DATA_PATH)
    result = df.filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id)).tail(count).collect()
    return result["sales"].to_list() if not result.is_empty() else [0.0] * count


def _parquet_context(item_id: str, store_id: str):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    df = pl.scan_parquet(PROCESSED_DATA_PATH)
    last_row = df.filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id)).tail(1).collect()
    return last_row.to_dicts()[0] if not last_row.is_empty() else None


def _parquet_context_at_d(item_id: str, store_id: str, d: int):
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    df = pl.scan_parquet(PROCESSED_DATA_PATH)
    row = df.filter(
        (pl.col("item_id") == item_id) & (pl.col("store_id") == store_id) & (pl.col("d") == d)
    ).collect()
    return row.to_dicts()[0] if not row.is_empty() else None


def get_history(item_id: str, store_id: str, count=84):
    if FEATURE_STORE == "redis":
        try:
            return redis_store.get_sales_history(store_id, item_id, count)
        except Exception:
            pass
    try:
        return _parquet_history(item_id, store_id, count)
    except Exception:
        return [0.0] * count


def get_item_context(item_id: str, store_id: str):
    if FEATURE_STORE == "redis":
        try:
            ctx = redis_store.get_ctx(store_id, item_id)
            if ctx:
                return ctx
        except Exception:
            pass
    try:
        return _parquet_context(item_id, store_id)
    except Exception:
        return None


def get_item_context_at_d(item_id: str, store_id: str, d: int):
    if FEATURE_STORE == "redis":
        try:
            ctx = redis_store.get_row_at_d(store_id, item_id, d)
            if ctx:
                return ctx
        except Exception:
            pass
    try:
        return _parquet_context_at_d(item_id, store_id, d)
    except Exception:
        return None


def fetch_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    df = pd.read_csv(LEADERBOARD_PATH).head(500)

    if "store_id" not in df.columns:
        try:
            meta_df = (
                pl.scan_parquet(PROCESSED_DATA_PATH)
                .select(["item_id", "store_id", "dept_id"])
                .unique()
                .collect()
                .to_pandas()
            )
            df = df.merge(meta_df, on="item_id", how="left")
        except Exception as e:
            print(f"Metadata Merge Failed: {e}")

    df["product_name"] = df["item_id"].map(PRODUCT_NAMES).fillna(df["item_id"])
    return df.fillna("N/A").to_dict(orient="records")
