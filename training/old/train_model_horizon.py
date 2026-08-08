import polars as pl
import lightgbm as lgb
import os
import gc
import json
import wandb
from pathlib import Path
from wandb.integration.lightgbm import wandb_callback, log_summary

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "backend" / "data" / "processed" / "m5_improved.parquet"
MODEL_DIR = BASE_DIR / "models" / "model_horizon"
os.makedirs(MODEL_DIR, exist_ok=True)

# Disk-Cache Paths
TRAIN_CSV = MODEL_DIR / "global_train_cache.csv"
VAL_CSV = MODEL_DIR / "global_val_cache.csv"

# 1. OPTIMIZED PARAMETERS
BEST_PARAMS = {
    'objective': 'quantile',
    'metric': 'quantile',
    'learning_rate': 0.05,
    'num_leaves': 128, 
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'num_threads': -1,
}

ALPHAS = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
HORIZONS = range(1, 29)

def build_disk_cache():
    """Phase 1: Build the 28x dataset on the hard drive to bypass RAM limits."""
    if TRAIN_CSV.exists() and VAL_CSV.exists():
        print("[skip] Disk cache already exists! Skipping Phase 1.")
        return

    print(f" Loading base data to build disk cache...")
    max_d = pl.read_parquet(DATA_PATH, columns=["d"])["d"].max()
    split_point = max_d - 28
    train_start_d = split_point - 365 

    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
        'wday', 'month', 'sell_price', 'price_norm',
        'roll_mean_7', 'roll_mean_28',
        'snap_CA', 'snap_TX', 'snap_WI',
        'horizon_day'
    ]
    cat_features = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    # We load the base 365-day dataset into RAM once
    df_base = (
        pl.read_parquet(DATA_PATH)
        .filter(pl.col("d") >= train_start_d)
        .sort(["store_id", "item_id", "d"])
    )

    print("[map] Building Translation Layer for LightGBM...")
    category_mappings = {}

    for col in cat_features:
        # Get unique string values and assign an integer ID to each
        unique_vals = df_base.select(col).unique().drop_nulls().to_series().to_list()
        col_map = {str(string_val): idx for idx, string_val in enumerate(unique_vals)}
        category_mappings[col] = col_map
        
        # Replace the strings with the integers in our temporary RAM dataframe
        df_base = df_base.with_columns(
            pl.col(col).cast(pl.String).replace(col_map).cast(pl.Int32).alias(col)
        )

    # Save the map to disk so we can translate LightGBM outputs back to strings later
    mapping_path = MODEL_DIR / "category_mappings.json"
    with open(mapping_path, "w") as f:
        json.dump(category_mappings, f, indent=4)
        
    print(f"[ok] Translation map saved to {mapping_path}")

    print(f"[write] Writing 28 Horizons directly to SSD... (This will take a few minutes)")
    
    # Open files in append mode
    with open(TRAIN_CSV, "a", newline='', encoding="utf-8") as f_train, open(VAL_CSV, "a", newline='', encoding="utf-8") as f_val:
        for h in HORIZONS:
            print(f"   -> Processing and appending Horizon {h}/28...")
            
            # Shift target and add horizon_day, keep it lazy until write
            df_h = df_base.with_columns([
                pl.col("sales").shift(-h).over(["store_id", "item_id"]).cast(pl.Float32).alias("target"),
                pl.lit(h).cast(pl.Int16).alias("horizon_day")
            ]).filter(pl.col("target").is_not_null())
            
            # Select target FIRST, then features
            columns_to_save = ["target"] + features
            
            train_chunk = df_h.filter(pl.col("d") <= split_point).select(columns_to_save)
            val_chunk = df_h.filter(pl.col("d") > split_point).select(columns_to_save)
            
            # Write to disk. Only write the header for the very first horizon!
            train_chunk.write_csv(f_train, include_header=(h == 1))
            val_chunk.write_csv(f_val, include_header=(h == 1))
            
            del df_h, train_chunk, val_chunk
            gc.collect()

    del df_base
    gc.collect()
    print("[ok] Disk cache successfully built!")

def train_out_of_core_suite():
    """Phase 2: Train LightGBM directly from the CSV files."""
    build_disk_cache()
    
    # FIX: Explicitly define the features list here in the training scope
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
        'wday', 'month', 'sell_price', 'price_norm',
        'roll_mean_7', 'roll_mean_28',
        'snap_CA', 'snap_TX', 'snap_WI',
        'horizon_day'
    ]
    
    cat_features = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    # Now this works perfectly
    cat_indices = [features.index(cat) for cat in cat_features]
    
    print("\n[init] Initializing LightGBM Out-of-Core Datasets...")
    # FIX: Tell LightGBM the target is strictly at column index 0
    lgb_params = {"header": True, "label_column": 0}
    
    lgb_train = lgb.Dataset(str(TRAIN_CSV), params=lgb_params, categorical_feature=cat_indices)
    lgb_eval = lgb.Dataset(str(VAL_CSV), reference=lgb_train, params=lgb_params)

    for alpha in ALPHAS:
        model_path = MODEL_DIR / f"global_model_alpha_{alpha}.txt"
        
        if model_path.exists():
            print(f"[skip] Skipping Alpha {alpha}, model already exists.")
            continue

        run_name = f"global_quantile_{alpha}"
        wandb.init(
            project="m5-forecasting-final", 
            name=run_name,
            config={**BEST_PARAMS, "alpha": alpha, "strategy": "global_out_of_core"},
            group="production_global_v2",
            reinit=True
        )

        print(f"\n[train] Training Global Quantile: {alpha} from Disk")
        current_params = BEST_PARAMS.copy()
        current_params['alpha'] = alpha
        
        model = lgb.train(
            current_params,
            lgb_train,
            valid_sets=[lgb_eval],
            valid_names=['valid'],
            num_boost_round=300, 
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
                wandb_callback()
            ]
        )
        
        log_summary(model)
        model.save_model(str(model_path))
        print(f"[ok] Saved model to {model_path}")
        wandb.finish()

if __name__ == "__main__":
    train_out_of_core_suite()