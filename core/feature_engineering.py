import polars as pl

PARTITION = ["store_id", "item_id"]

HORIZON_BASE_FEATURES = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "wday", "month", "sell_price", "price_norm",
    "snap_CA", "snap_TX", "snap_WI",
    "price_momentum_7d", "price_momentum_28d",
    "lag_28", "roll_mean_7_lag_28", "roll_mean_28_lag_28",
    "masked_roll_mean_28_lag_28", "ema_lag_28", "days_since_last_sale_lag_28",
]

HORIZON_TRAIN_FEATURES = HORIZON_BASE_FEATURES + ["horizon_day", "target_wday"]
HORIZON_CAT_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "target_wday"]

REDIS_ROW_COLUMNS = ["d", "sales"] + HORIZON_BASE_FEATURES


def compute_target_wday(wday: int, horizon_day: int) -> int:
    return ((int(wday) + int(horizon_day) - 1) % 7) + 1


def horizon_feature_engineer(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Notebook-aligned horizon features with lag-28 rolling means."""
    df = lazy_df.with_columns([
        (pl.col("sell_price") / pl.col("sell_price").shift(7).over(PARTITION)).alias("price_momentum_7d"),
        (pl.col("sell_price") / pl.col("sell_price").shift(28).over(PARTITION)).alias("price_momentum_28d"),
        pl.col("sales").shift(28).over(PARTITION).alias("lag_28"),
        (pl.col("sell_price") / pl.col("sell_price").max().over(PARTITION)).alias("price_norm"),
    ]).with_columns(
        pl.int_range(1, pl.len() + 1).over(PARTITION).alias("row_nr")
    ).with_columns([
        pl.col("lag_28").rolling_mean(window_size=7).over(PARTITION).alias("roll_mean_7_lag_28"),
        pl.col("lag_28").rolling_mean(window_size=28).over(PARTITION).alias("roll_mean_28_lag_28"),
        pl.when(pl.col("lag_28") > 0).then(pl.col("lag_28")).otherwise(None)
        .rolling_mean(window_size=28, min_periods=1).forward_fill()
        .over(PARTITION).alias("masked_roll_mean_28_lag_28"),
        pl.col("lag_28").ewm_mean(alpha=0.1, ignore_nulls=True).over(PARTITION).alias("ema_lag_28"),
        pl.when(pl.col("lag_28") > 0).then(pl.col("row_nr")).otherwise(None)
        .forward_fill().over(PARTITION).alias("last_sale_row"),
    ]).with_columns([
        (pl.col("row_nr") - pl.col("last_sale_row")).fill_null(0).alias("days_since_last_sale_lag_28"),
    ]).drop(["row_nr", "last_sale_row"]).filter(
        pl.col("roll_mean_28_lag_28").is_not_null() & pl.col("price_momentum_28d").is_not_null()
    )
    return df


def feature_engineer(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Default production path: horizon notebook feature set."""
    return horizon_feature_engineer(lazy_df)
