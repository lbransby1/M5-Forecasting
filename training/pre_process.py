import polars as pl
import os

# def preprocess_m5():
#     print("🚀 Starting High-Performance Preprocessing...")
    
#     # 1. Load data with Polars
#     # We use scan_csv (Lazy) so Polars can optimize the query plan before execution
#     sales = pl.read_csv("training/data/raw/sales_train_evaluation.csv")
#     calendar = pl.read_csv("training/data/raw/calendar.csv")
#     prices = pl.read_csv("training/data/raw/sell_prices.csv")

#     # 2. MELT: Turn d_1...d_1941 into a 'day' and 'sales' column
#     id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
#     print("Melting 58M rows...")
#     sales_long = sales.unpivot(
#         index=id_vars, 
#         variable_name="d", 
#         value_name="sales"
#     )

#     # 3. FEATURE ENGINEERING: Basic Lags & Rolling Means
#     # We do this here so they are baked into the Parquet file
#     print("Generating temporal features (Lags & Rolling)...")
#     df = (
#         sales_long
#         .join(calendar, on="d", how="left")
#         .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
#     )

#     # 4. MEMORY OPTIMIZATION: Categoricals
#     # Turning strings into integers (categories) drops memory usage by ~80%
#     print("Optimizing memory with Categoricals...")
#     cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]
#     df = df.with_columns([
#         pl.col(c).cast(pl.Categorical) for c in cat_cols
#     ])

#     # 5. SAVE AS PARQUET
#     os.makedirs("training/data/processed", exist_ok=True)
#     df.write_parquet("training/data/processed/m5_full.parquet")
#     print("✅ Preprocessing complete! Saved to training/data/processed/m5_full.parquet")

# if __name__ == "__main__":
#     preprocess_m5()
import os
import polars as pl
from pathlib import Path

# BASE_DIR is C:\M5 (if your script is in C:\M5\training)
BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_m5(add_features=False):
    # 1. FIX: Pathlib handles the slashes. No quotes around BASE_DIR.
    processed_dir = BASE_DIR / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"🚀 Preprocessing: {'WITH' if add_features else 'WITHOUT'} feature engineering...")

    # 2. FIX: BASE_DIR / "string" (Don't start the string with a slash!)
    sales = pl.scan_csv(BASE_DIR / "data/raw/sales_train_evaluation.csv")
    calendar = pl.scan_csv(BASE_DIR / "data/raw/calendar.csv")
    prices = pl.scan_csv(BASE_DIR / "data/raw/sell_prices.csv")

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    pipeline = (
        sales.unpivot(index=id_vars, variable_name="d", value_name="sales")
        .join(calendar, on="d", how="left")
        .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .with_columns([
            pl.col("sales").cast(pl.Int16),
            pl.col("sell_price").cast(pl.Float32),
            pl.col("d").str.replace("d_", "").cast(pl.Int16),
            *[pl.col(c).cast(pl.Categorical) for c in id_vars if c != "id"]
        ])
    )

    if add_features:
        print("🛠 Adding Rolling Windows... this takes a minute...")
        # NOTE: Grouping by ["item_id", "store_id"] is more accurate for M5 than just item_id
        pipeline = pipeline.with_columns([
            pl.col("sales").shift(28).rolling_mean(window_size=7).over(["item_id", "store_id"]).alias("roll_mean_7"),
            pl.col("sales").shift(28).rolling_mean(window_size=28).over(["item_id", "store_id"]).alias("roll_mean_28"),
            (pl.col("sell_price") / pl.col("sell_price").max().over(["item_id", "store_id"])).alias("price_norm")
        ])
        pipeline = pipeline.filter(pl.col("roll_mean_28").is_not_null())

    filename = "m5_improved.parquet" if add_features else "m5_baseline.parquet"
    path = processed_dir / filename # Pathlib handles this perfectly

    # 3. Execution
    df = pipeline.collect()
    df.write_parquet(path)

    print(f"✅ Created: {path}")
    print(f"📊 Total Rows Generated: {df.height:,}")
    return path

if __name__ == "__main__":
    preprocess_m5(add_features=True)