# backend/core/data.py
import os
import json
import pandas as pd
import polars as pl

# --- CONFIGURATION ---
DATA_DIR = "backend/data"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed/m5_improved.parquet"
LEADERBOARD_PATH = f"{DATA_DIR}/item_leaderboard.csv"
MAPPING_PATH = "backend/category_mappings.json"
PRODUCT_MAP_PATH = "backend/product_map.json"

# --- GLOBAL ASSETS ---
CALENDAR = pd.read_csv(f"{DATA_DIR}/raw/calendar.csv")
CALENDAR['d_num'] = CALENDAR['d'].str.replace('d_', '').astype(int)
MAPPINGS = {}
PRODUCT_NAMES = {}

def load_data_assets():
    global MAPPINGS, PRODUCT_NAMES
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as f:
            MAPPINGS = json.load(f)
    if os.path.exists(PRODUCT_MAP_PATH):
        with open(PRODUCT_MAP_PATH, "r") as f:
            PRODUCT_NAMES = json.load(f)
    print(f"✅ Data assets loaded: {len(PRODUCT_NAMES)} product names.")

def get_history(item_id: str, store_id: str, count=84):
    if not os.path.exists(PROCESSED_DATA_PATH): return [0.0] * count
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        result = df.filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id)).tail(count).collect()
        return result["sales"].to_list() if not result.is_empty() else [0.0] * count
    except: return [0.0] * count

def get_item_context(item_id: str, store_id: str):
    if not os.path.exists(PROCESSED_DATA_PATH): return None
    try:
        df = pl.scan_parquet(PROCESSED_DATA_PATH)
        last_row = df.filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id)).tail(1).collect()
        return last_row.to_dicts()[0] if not last_row.is_empty() else None
    except: return None

def fetch_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH): return []
    df = pd.read_csv(LEADERBOARD_PATH).head(500)
    
    if 'store_id' not in df.columns:
        try:
            meta_df = pl.scan_parquet(PROCESSED_DATA_PATH).select(["item_id", "store_id", "dept_id"]).unique().collect().to_pandas()
            df = df.merge(meta_df, on="item_id", how="left")
        except Exception as e:
            print(f"Metadata Merge Failed: {e}")
            
    df['product_name'] = df['item_id'].map(PRODUCT_NAMES).fillna(df['item_id'])
    return df.fillna("N/A").to_dict(orient="records")