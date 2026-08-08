import gc
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl
import wandb
from wandb.integration.lightgbm import log_summary, wandb_callback

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.feature_engineering import HORIZON_TRAIN_FEATURES, compute_target_wday

DATA_PATH = BASE_DIR / "backend" / "data" / "processed" / "m5_improved.parquet"
MODEL_DIR = BASE_DIR / "models" / "model_horizon_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = MODEL_DIR / "global_train_cache.csv"
VAL_CSV = MODEL_DIR / "global_val_cache.csv"

TRAIN_BIN = MODEL_DIR / "global_train_cache.bin"
VAL_BIN = MODEL_DIR / "global_val_cache.bin"
TRAIN_WINDOW_DAYS = int(os.environ.get("TRAIN_WINDOW_DAYS", "365"))

# Windows + huge out-of-core CSVs can crash LightGBM with access violations.
# Single-thread + column-wise is much more stable.
if os.name == "nt":
    BEST_PARAMS = {
        "objective": "quantile",
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": 1,
        "num_threads": 4,
        "force_col_wise": True,
    }
else:
    BEST_PARAMS = {
        "objective": "quantile",
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": 1,
        "num_threads": -1,
    }

ALPHAS = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
HORIZONS = range(1, 29)
CAT_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "target_wday"]


def build_disk_cache(force: bool = False):
    if TRAIN_CSV.exists() and VAL_CSV.exists() and not force:
        print("[skip] Disk cache already exists.")
        return

    if force:
        TRAIN_CSV.unlink(missing_ok=True)
        VAL_CSV.unlink(missing_ok=True)
        TRAIN_BIN.unlink(missing_ok=True)
        VAL_BIN.unlink(missing_ok=True)

    print("[load] Building v2 horizon disk cache...")
    max_d = pl.read_parquet(DATA_PATH, columns=["d"])["d"].max()
    split_point = max_d - 28
    train_start_d = split_point - TRAIN_WINDOW_DAYS
    print(f"[load] Training window: {TRAIN_WINDOW_DAYS} days (set TRAIN_WINDOW_DAYS to change)")

    features = HORIZON_TRAIN_FEATURES
    df_base = (
        pl.read_parquet(DATA_PATH)
        .filter(pl.col("d") >= train_start_d)
        .sort(["store_id", "item_id", "d"])
    )

    category_mappings = {}
    for col in CAT_FEATURES:
        if col == "target_wday":
            continue
        unique_vals = df_base.select(col).unique().drop_nulls().to_series().to_list()
        category_mappings[col] = {str(v): idx for idx, v in enumerate(unique_vals)}
        df_base = df_base.with_columns(
            pl.col(col).cast(pl.String).replace(category_mappings[col]).cast(pl.Int32).alias(col)
        )

    category_mappings["target_wday"] = {str(i): i - 1 for i in range(1, 8)}

    mapping_path = MODEL_DIR / "category_mappings.json"
    with open(mapping_path, "w") as f:
        json.dump(category_mappings, f, indent=4)
    print(f"[ok] Saved mappings to {mapping_path}")

    with open(TRAIN_CSV, "a", newline="", encoding="utf-8") as f_train, open(
        VAL_CSV, "a", newline="", encoding="utf-8"
    ) as f_val:
        for h in HORIZONS:
            print(f"   -> horizon {h}/28")
            df_h = df_base.with_columns([
                pl.col("sales").shift(-h).over(["store_id", "item_id"]).cast(pl.Float32).alias("target"),
                pl.lit(h).cast(pl.Int16).alias("horizon_day"),
                (((pl.col("wday") + h - 1) % 7) + 1).cast(pl.Int8).alias("target_wday"),
            ]).filter(pl.col("target").is_not_null())

            df_h = df_h.with_columns(
                pl.col("target_wday").cast(pl.String).replace(category_mappings["target_wday"]).cast(pl.Int32)
            )

            columns_to_save = ["target"] + features
            train_chunk = df_h.filter(pl.col("d") <= split_point).select(columns_to_save)
            val_chunk = df_h.filter(pl.col("d") > split_point).select(columns_to_save)
            train_chunk.write_csv(f_train, include_header=(h == 1))
            val_chunk.write_csv(f_val, include_header=(h == 1))
            del df_h, train_chunk, val_chunk
            gc.collect()

    del df_base
    gc.collect()
    print("[ok] Disk cache built.")


def _load_datasets(cat_indices):
    lgb_params = {"header": True, "label_column": 0}

    if TRAIN_BIN.exists() and VAL_BIN.exists():
        print(f"[init] Loading LightGBM binary cache...")
        print(f"         train: {TRAIN_BIN}")
        print(f"         valid: {VAL_BIN}")
        lgb_train = lgb.Dataset(str(TRAIN_BIN))
        lgb_eval = lgb.Dataset(str(VAL_BIN), reference=lgb_train)
        print("[ok] Binary datasets loaded.")
        return lgb_train, lgb_eval

    print("[init] Loading out-of-core LightGBM datasets from CSV...")
    print(f"         train: {TRAIN_CSV}")
    print(f"         valid: {VAL_CSV}")
    print("         (First CSV load can take 10-30+ minutes with no output — this is normal.)")
    lgb_train = lgb.Dataset(str(TRAIN_CSV), params=lgb_params, categorical_feature=cat_indices)
    lgb_eval = lgb.Dataset(str(VAL_CSV), reference=lgb_train, params=lgb_params)
    print("[init] Saving binary cache for faster/safer restarts...")
    lgb_train.save_binary(str(TRAIN_BIN))
    lgb_eval.save_binary(str(VAL_BIN))
    print("[ok] Datasets loaded and binary cache written.")
    return lgb_train, lgb_eval


def train_out_of_core_suite(force_cache: bool = False):
    build_disk_cache(force=force_cache)

    cat_indices = [HORIZON_TRAIN_FEATURES.index(cat) for cat in CAT_FEATURES if cat in HORIZON_TRAIN_FEATURES]
    lgb_train, lgb_eval = _load_datasets(cat_indices)

    for alpha in ALPHAS:
        model_path = MODEL_DIR / f"global_model_alpha_{alpha}.txt"
        if model_path.exists():
            print(f"[skip] alpha {alpha}")
            continue

        print(f"\n[train] alpha={alpha} — initializing WandB run...")
        run = wandb.init(
            project="m5-forecasting-final",
            name=f"global_quantile_v2_{alpha}",
            config={**BEST_PARAMS, "alpha": alpha, "strategy": "horizon_v2"},
            group="production_global_v3",
            reinit="finish_previous",
        )
        print(f"[train] WandB run: {run.url}")

        params = {**BEST_PARAMS, "alpha": alpha}
        print(f"[train] LightGBM boosting for alpha={alpha} (logs every 10 rounds)...")
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_eval],
            valid_names=["valid"],
            num_boost_round=50,
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, min_delta=0.0001),
                lgb.log_evaluation(period=10),
                wandb_callback(),
            ],
        )
        log_summary(model)
        model.save_model(str(model_path))
        print(f"[ok] saved {model_path}")
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    train_out_of_core_suite(force_cache=args.force_cache)
