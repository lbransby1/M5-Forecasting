import polars as pl
import lightgbm as lgb
import os
import gc
import wandb
from pathlib import Path
from wandb.integration.lightgbm import wandb_callback, log_summary

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "backend"/ "data" / "processed" / "m5_improved.parquet"
MODEL_DIR = BASE_DIR / "models" / "model_alpha"
os.makedirs(MODEL_DIR, exist_ok=True)

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

def train_production_suite():
    print(f"📂 Loading data from {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
        'wday', 'month', 'sell_price', 'price_norm',
        'roll_mean_7', 'roll_mean_28',
        'snap_CA', 'snap_TX', 'snap_WI'
    ]
    
    cat_features = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    split_point = df["d"].max() - 28
    
    # Pre-split and convert to avoid memory spikes
    X_train = df.filter(pl.col("d") <= split_point).select(features).to_pandas()
    y_train = df.filter(pl.col("d") <= split_point).select("sales").to_series().to_numpy()
    X_val = df.filter(pl.col("d") > split_point).select(features).to_pandas()
    y_val = df.filter(pl.col("d") > split_point).select("sales").to_series().to_numpy()

    del df
    gc.collect()

    # Create Base Datasets once to reuse across alphas
    lgb_train_base = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=True)
    lgb_eval_base = lgb.Dataset(X_val, y_val, reference=lgb_train_base, free_raw_data=True)

    del X_train, y_train, X_val, y_val
    gc.collect()

    for alpha in ALPHAS:
        model_path = MODEL_DIR / f"model_alpha_{alpha}.txt"
        
        # SKIP if already trained
        if model_path.exists():
            print(f"⏩ Skipping Alpha {alpha}, model already exists.")
            continue

        run_name = f"quantile_{alpha}"
        wandb.init(
            project="m5-forecasting-final", 
            name=run_name,
            config={**BEST_PARAMS, "alpha": alpha},
            group="production_v1",
            reinit=True
        )

        print(f"\n🎯 Training Quantile: {alpha}")
        current_params = BEST_PARAMS.copy()
        current_params['alpha'] = alpha
        
        # REDUCED num_boost_round to 200 based on plateau observation
        model = lgb.train(
            current_params,
            lgb_train_base,
            valid_sets=[lgb_eval_base],
            valid_names=['valid'],
            num_boost_round=200, 
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
                wandb_callback()
            ]
        )
        
        log_summary(model)
        model.save_model(str(model_path))
        print(f"✅ Saved model to {model_path}")
        wandb.finish()

if __name__ == "__main__":
    train_production_suite()