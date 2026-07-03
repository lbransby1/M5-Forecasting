import os
import polars as pl
import argparse

# 1. Setup S3 storage options for Polars
storage_options = {
    "endpoint_url": os.getenv("STORAGE_ENDPOINT_URL"),
    "key": os.getenv("STORAGE_ACCESS_KEY_ID"),
    "secret": os.getenv("STORAGE_SECRET_ACCESS_KEY"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--s3_raw_dir", type=str, default="s3://efficient-snackbox-4rjagp/raw")
parser.add_argument("--s3_output_path", type=str, default="s3://efficient-snackbox-4rjagp/processed/m5_improved.parquet")
args = parser.parse_args()

def preprocess_m5(add_features=True):
    print(f"🚀 Scaling Preprocessing via Store-by-Store Loop (S3 Enabled)...")

    # Load small tables directly from S3
    calendar = pl.read_csv(f"{args.s3_raw_dir}/calendar.csv", storage_options=storage_options).with_columns([
        pl.col("d").str.replace("d_", "").cast(pl.Int16)
    ])

    prices = pl.read_csv(f"{args.s3_raw_dir}/sell_prices.csv", storage_options=storage_options).with_columns([
        pl.col("store_id").cast(pl.Categorical),
        pl.col("item_id").cast(pl.Categorical)
    ])

    sales_lazy = pl.scan_csv(f"{args.s3_raw_dir}/sales_train_evaluation.csv", storage_options=storage_options)
    
    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    store_files = []

    for store in stores:
        print(f"📦 Processing Store: {store}...")
        
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
            .sort(["item_id", "d"]) 
        )

        if add_features:
            store_pipeline = store_pipeline.with_columns([
                pl.col("sales").shift(28).rolling_mean(7).over("item_id").alias("roll_mean_7"),
                pl.col("sales").shift(28).rolling_mean(28).over("item_id").alias("roll_mean_28"),
                (pl.col("sell_price") / pl.col("sell_price").max().over("item_id")).alias("price_norm")
            ]).filter(pl.col("roll_mean_28").is_not_null())

        # Write directly to S3
        temp_s3_path = f"{args.s3_output_path.rsplit('/', 1)[0]}/temp_{store}.parquet"
        store_pipeline.collect().write_parquet(temp_s3_path, storage_options=storage_options)
        store_files.append(temp_s3_path)

    print("🔗 Stitching all stores directly in S3...")
    pl.concat([pl.scan_parquet(f, storage_options=storage_options) for f in store_files]).sink_parquet(
        args.s3_output_path, storage_options=storage_options
    )
    
    # Cleanup: Delete temp files from S3
    import s3fs
    fs = s3fs.S3FileSystem(**storage_options)
    for f in store_files:
        fs.rm(f)

    print(f"✅ SUCCESS: {args.s3_output_path}")

if __name__ == "__main__":
    preprocess_m5()