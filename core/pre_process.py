import sys
import os
import argparse
import polars as pl

# 1. Path routing for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from core import feature_engineering

RAW_DIR_CANDIDATES = [
    "backend/data/raw",
    "data/raw",
]


def resolve_raw_dir(raw_dir: str | None = None) -> str:
    if raw_dir:
        calendar = os.path.join(raw_dir, "calendar.csv")
        if os.path.exists(calendar):
            return raw_dir
        raise FileNotFoundError(
            f"calendar.csv not found in {raw_dir}. "
            f"Download data with: python core/download_data.py --output_dir backend/data/raw"
        )

    for candidate in RAW_DIR_CANDIDATES:
        calendar = os.path.join(candidate, "calendar.csv")
        if os.path.exists(calendar):
            print(f"Using raw data directory: {candidate}", file=sys.stderr)
            return candidate

    raise FileNotFoundError(
        "No raw M5 CSVs found. Run: python core/download_data.py --output_dir backend/data/raw"
    )

# ==========================================
# CORE PIPELINE LOGIC
# ==========================================
def preprocess_m5(
    mode="local", 
    raw_dir=None, 
    output_path="backend/data/processed/m5_improved.parquet", 
    add_features=True
):
    """
    Main preprocessing pipeline. Can be called directly via Python or CLI.
    """
    raw_dir = resolve_raw_dir(raw_dir)
    output_path = os.path.abspath(output_path)
    print(f"Scaling Preprocessing via Store-by-Store Loop ({mode.upper()} Mode)...")

    # 1. MODE CONFIGURATION (Moved inside the function)
    use_s3 = (mode == "s3")

    if use_s3:
        storage_options = {
            "endpoint_url": os.getenv("STORAGE_ENDPOINT_URL"),
            "key": os.getenv("STORAGE_ACCESS_KEY_ID"),
            "secret": os.getenv("STORAGE_SECRET_ACCESS_KEY"),
        }
        storage_kwargs = {"storage_options": storage_options} 
    else:
        storage_options = None
        storage_kwargs = {} 
        # Crucial for local: ensure the output folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. LOAD DATA
    calendar = pl.read_csv(f"{raw_dir}/calendar.csv", **storage_kwargs).with_columns([
        pl.col("d").str.replace("d_", "").cast(pl.Int16)
    ])

    prices = pl.read_csv(f"{raw_dir}/sell_prices.csv", **storage_kwargs).with_columns([
        pl.col("store_id").cast(pl.Categorical),
        pl.col("item_id").cast(pl.Categorical)
    ])

    sales_lazy = pl.scan_csv(f"{raw_dir}/sales_train_evaluation.csv", **storage_kwargs)
    
    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    store_files = []

    # 3. PROCESS STORES
    for store in stores:
        print(f"Processing Store: {store}...")
        
        store_prices = prices.lazy().filter(pl.col("store_id") == store)
        
        store_pipeline = (
            sales_lazy.filter(pl.col("store_id") == store)
            .unpivot(index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], 
                     variable_name="d", value_name="sales")
            .with_columns([
                pl.col("d").str.replace("d_", "").cast(pl.Int16),
                pl.col("sales").cast(pl.Int16),
                *[pl.col(c).cast(pl.Categorical) for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]]
            ])
            .join(calendar.lazy(), on="d", how="left")
            .join(store_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
            .sort(["store_id", "item_id", "d"])
        )

        if add_features:
            store_pipeline = feature_engineering.horizon_feature_engineer(store_pipeline)

        # Handle temp path formatting depending on mode
        if use_s3:
            base_dir = output_path.rsplit('/', 1)[0]
            temp_path = f"{base_dir}/temp_{store}.parquet"
        else:
            base_dir = os.path.dirname(output_path)
            os.makedirs(base_dir, exist_ok=True)
            temp_path = os.path.abspath(os.path.join(base_dir, f"temp_{store}.parquet"))

        if os.path.exists(temp_path):
            print(f"   -> Reusing existing temp file for {store}")
        else:
            print(f"   -> Writing {temp_path}")
            store_pipeline.collect().write_parquet(temp_path, **storage_kwargs)

        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Expected temp parquet was not created: {temp_path}")

        store_files.append(temp_path)

    # 4. STITCH FILES
    missing = [f for f in store_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing temp parquet files before stitch: {missing}")

    print("Stitching all stores...")
    stitched = pl.concat([pl.read_parquet(f, **storage_kwargs) for f in store_files])
    stitched.write_parquet(output_path, **storage_kwargs)
    
    # 5. CLEANUP
    print("Cleaning up temp files...")
    if use_s3:
        import s3fs
        fs = s3fs.S3FileSystem(**storage_options)
        for f in store_files:
            fs.rm(f)
    else:
        for f in store_files:
            if os.path.exists(f):
                os.remove(f)

    print(f"SUCCESS: Saved to {output_path}\n{'='*35}\n")
    return output_path


# ==========================================
# COMMAND LINE INTERFACE (CLI)
# ==========================================
if __name__ == "__main__":
    # This block ONLY runs if you type `python preprocess.py` in the terminal.
    # It is completely ignored if you import the file elsewhere.
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["local", "s3"], default="local")
    parser.add_argument("--raw_dir", type=str, default=None, help="Raw CSV directory (auto-detects data/raw or backend/data/raw)")
    parser.add_argument("--output_path", type=str, default="backend/data/processed/m5_improved.parquet")
    parser.add_argument("--no_features", action="store_true", help="Pass this flag to disable feature engineering")
    args = parser.parse_args()

    preprocess_m5(
        mode=args.mode,
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        add_features=not args.no_features
    )