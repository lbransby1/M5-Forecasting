import os
import polars as pl
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="backend/data/raw") 
parser.add_argument("--output_file", type=str, default="backend/data/processed/m5_improved.parquet")
parser.add_argument("--add_features", action="store_true", default=True) 
args = parser.parse_args()

RAW_DATA_DIR = Path(args.input_dir)
OUTPUT_PATH = Path(args.output_file)

def preprocess_m5(add_features=True):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Scaling Preprocessing via Store-by-Store Loop...")

    # 1. Load small tables into memory (They are tiny)
    calendar = pl.read_csv(RAW_DATA_DIR / "calendar.csv").with_columns([
        pl.col("d").str.replace("d_", "").cast(pl.Int16)
    ])

    prices = pl.read_csv(RAW_DATA_DIR / "sell_prices.csv").with_columns([
        pl.col("store_id").cast(pl.Categorical),
        pl.col("item_id").cast(pl.Categorical)
    ])

    # 2. Scan the big sales file
    sales_lazy = pl.scan_csv(RAW_DATA_DIR / "sales_train_evaluation.csv")
    
    # Get unique stores
    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    
    store_files = []

    for store in stores:
        print(f"📦 Processing Store: {store}...")
        
        # 1. Pre-filter prices to drastically speed up the join
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
            # 2. CRITICAL: Sort chronologically before doing rolling math!
            .sort(["item_id", "d"]) 
        )

        if add_features:
            store_pipeline = store_pipeline.with_columns([
                pl.col("sales").shift(28).rolling_mean(7).over("item_id").alias("roll_mean_7"),
                pl.col("sales").shift(28).rolling_mean(28).over("item_id").alias("roll_mean_28"),
                (pl.col("sell_price") / pl.col("sell_price").max().over("item_id")).alias("price_norm")
            ]).filter(pl.col("roll_mean_28").is_not_null())

        # Save this store to a temp file
        temp_path = f"backend/data/processed/temp_{store}.parquet"
        
        # 3. CRITICAL FIX: Use .collect().write_parquet() instead of .sink_parquet()
        store_pipeline.collect().write_parquet(temp_path)
        store_files.append(temp_path)

        # 4. Final Step: Combine all store parquets into one
    print("🔗 Stitching all stores into final production parquet...")
        
        # We can still use sink_parquet here because it is just a dumb concatenation (no window functions)
    pl.concat([pl.scan_parquet(f) for f in store_files]).sink_parquet(OUTPUT_PATH)
    
    # Clean up temp files
    for f in store_files: os.remove(f)

    print(f"✅ SUCCESS: {OUTPUT_PATH}")
    return OUTPUT_PATH

if __name__ == "__main__":
    preprocess_m5(add_features=args.add_features)