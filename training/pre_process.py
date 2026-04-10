import os
import polars as pl
import argparse
from pathlib import Path

# 1. SETUP ARGUMENTS (This is what run_pipeline.sh talks to)
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="backend/data/raw") 
parser.add_argument("--output_file", type=str, default="backend/data/processed/m5_improved.parquet")
parser.add_argument("--add_features", action="store_true", default=True) 
args = parser.parse_args()

# Define Paths
RAW_DATA_DIR = Path(args.input_dir)
OUTPUT_PATH = Path(args.output_file)

def preprocess_m5(add_features=True):
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Preprocessing: {'WITH' if add_features else 'WITHOUT'} feature engineering...")
    print(f"📂 Reading from: {RAW_DATA_DIR}")

    # 2. LAZY LOADING & KEY CLEANING
    # We clean 'd' (remove 'd_') immediately to join on efficient Integers (Int16)
    sales = pl.scan_csv(RAW_DATA_DIR / "sales_train_evaluation.csv")
    
    calendar = pl.scan_csv(RAW_DATA_DIR / "calendar.csv").with_columns([
        pl.col("d").str.replace("d_", "").cast(pl.Int16)
    ])

    prices = pl.scan_csv(RAW_DATA_DIR / "sell_prices.csv").with_columns([
        pl.col("store_id").cast(pl.Categorical),
        pl.col("item_id").cast(pl.Categorical)
    ])

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # 3. BUILD PIPELINE
    pipeline = (
        sales.unpivot(index=id_vars, variable_name="d", value_name="sales")
        # Optimization: Clean 'd' and cast Categories BEFORE joins
        .with_columns([
            pl.col("d").str.replace("d_", "").cast(pl.Int16),
            pl.col("sales").cast(pl.Int16),
            *[pl.col(c).cast(pl.Categorical) for c in id_vars if c != "id"]
        ])
        .join(calendar, on="d", how="left")
        .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .with_columns([
            pl.col("sell_price").cast(pl.Float32),
        ])
    )

    if add_features:
        print("🛠 Adding Rolling Windows and Price Elasticity...")
        pipeline = pipeline.with_columns([
            # Shift 28 ensures no data leakage from the future 
            pl.col("sales").shift(28).rolling_mean(window_size=7).over(["item_id", "store_id"]).alias("roll_mean_7"),
            pl.col("sales").shift(28).rolling_mean(window_size=28).over(["item_id", "store_id"]).alias("roll_mean_28"),
            # Normalized Price
            (pl.col("sell_price") / pl.col("sell_price").max().over(["item_id", "store_id"])).alias("price_norm")
        ])
        # Drop nulls from rolling window warmup
        pipeline = pipeline.filter(pl.col("roll_mean_28").is_not_null())

    # 4. EXECUTION (Streaming mode to save Railway RAM)
    print(f"⏳ Streaming data to Parquet... {OUTPUT_PATH}")
    
    # We use sink_parquet for 58M rows because collect() will likely 
    # exceed Railway's RAM limits.
    pipeline.sink_parquet(OUTPUT_PATH)

    print(f"✅ Created: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    preprocess_m5(add_features=args.add_features)