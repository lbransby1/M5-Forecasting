import os
import polars as pl
import argparse
from pathlib import Path

# 1. SETUP ARGUMENTS (This is what run_pipeline.sh talks to)
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="data/raw") # Default to local for dev
parser.add_argument("--output_file", type=str, default="data/processed/m5_improved.parquet")
parser.add_argument("--add_features", action="store_true", default=True) # Toggle via CLI
args = parser.parse_args()

# Define Paths
RAW_DATA_DIR = Path(args.input_dir)
OUTPUT_PATH = Path(args.output_file)

def preprocess_m5(add_features=True):
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Preprocessing: {'WITH' if add_features else 'WITHOUT'} feature engineering...")
    print(f"📂 Reading from: {RAW_DATA_DIR}")

    # 2. LAZY LOADING (Scan instead of Read)
    sales = pl.scan_csv(RAW_DATA_DIR / "sales_train_evaluation.csv")
    # calendar = pl.scan_csv(RAW_DATA_DIR / "calendar.csv")
    # prices = pl.scan_csv(RAW_DATA_DIR / "sell_prices.csv")

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # 3. BUILD PIPELINE
    calendar = pl.scan_csv(RAW_DATA_DIR / "calendar.csv").with_columns([
    pl.col("d").cast(pl.Categorical) # d needs to be categorical to match sales
])

    prices = pl.scan_csv(RAW_DATA_DIR / "sell_prices.csv").with_columns([
    pl.col("store_id").cast(pl.Categorical), # MUST match sales store_id type
    pl.col("item_id").cast(pl.Categorical)   # MUST match sales item_id type
])

# 2. Build the main pipeline
    pipeline = (
    sales.unpivot(index=id_vars, variable_name="d", value_name="sales")
    .with_columns([
        pl.col(c).cast(pl.Categorical) for c in id_vars if c != "id"
    ])
    .with_columns(pl.col("d").cast(pl.Categorical)) # Ensure 'd' is categorical here too
    .join(calendar, on="d", how="left")
    .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .with_columns([
            pl.col("sales").cast(pl.Int16),
            pl.col("sell_price").cast(pl.Float32),
            pl.col("d").str.replace("d_", "").cast(pl.Int16),
        ])
    )

    if add_features:
        print("🛠 Adding Rolling Windows and Price Elasticity...")
        pipeline = pipeline.with_columns([
            # Shift 28 ensures no data leakage from the future 
            pl.col("sales").shift(28).rolling_mean(window_size=7).over(["item_id", "store_id"]).alias("roll_mean_7"),
            pl.col("sales").shift(28).rolling_mean(window_size=28).over(["item_id", "store_id"]).alias("roll_mean_28"),
            # Normalized Price (Price Elasticity Feature)
            (pl.col("sell_price") / pl.col("sell_price").max().over(["item_id", "store_id"])).alias("price_norm")
        ])
        # Drop rows where we don't have enough history for the 28-day roll
        pipeline = pipeline.filter(pl.col("roll_mean_28").is_not_null())

    # 4. EXECUTION (The only time data is actually moved)
    print("⏳ Collecting data and writing to Parquet... (This will use significant RAM)")
    df = pipeline.collect()
    df.write_parquet(OUTPUT_PATH)

    print(f"✅ Created: {OUTPUT_PATH}")
    print(f"📊 Total Rows Generated: {df.height:,}")
    return OUTPUT_PATH

if __name__ == "__main__":
    preprocess_m5(add_features=args.add_features)