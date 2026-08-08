import sys
import os
import argparse
import importlib.util
import gc
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Path routing for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from core import pre_process

# ==========================================
# GLOBAL M5 CONFIGURATION
# ==========================================
END_TRAIN_DAY = 1913
FORECAST_HORIZON = 28
N_FOLDS = 3
TARGET_STORE = "ALL" 

def load_external_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'run_model'):
        raise AttributeError(f"The file {file_path} must contain a 'run_model' function.")
    return module

def get_cv_splits(n_folds, end_day, horizon):
    splits = []
    for fold in range(n_folds):
        val_end = end_day - (fold * horizon)
        val_start = val_end - horizon + 1
        train_end = val_start - 1
        splits.append({"fold": fold + 1, "train_end": train_end, "val_start": val_start, "val_end": val_end})
    return splits

def wspl_metric(preds, labels, weights, alpha=0.5):
    error = labels - preds
    pinball_loss = np.where(error >= 0, alpha * error, (alpha - 1) * error)
    return np.mean(pinball_loss * weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 Dynamic Model Trainer")
    parser.add_argument("--model", type=str, required=True, help="Path to the model .py file")
    parser.add_argument("--data_path", type=str, default="data/processed/m5-processed.parquet")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--force_preprocess", action="store_true", help="Force data preprocessing")
    args = parser.parse_args()

    # 1. Preprocessing Check
    if args.force_preprocess or not os.path.exists(args.data_path):
        trigger = "Force flag detected" if args.force_preprocess else "Data not found"
        print(f"📦 {trigger}. Running preprocessor...")
        pre_process.preprocess_m5(mode="local", raw_dir=args.raw_dir, output_path=args.data_path, add_features=True)

    # 2. Load Model
    print(f"\n{'='*60}\nLoading custom model from: {args.model}\n{'='*60}")
    custom_model = load_external_model(args.model)

    # 3. Setup Schema
    lazy_df = pl.scan_parquet(args.data_path)
    all_cols = lazy_df.collect_schema().names()
    
    exclude_cols = ['id', 'd', 'date', 'sales', 'wm_yr_wk', 'weight', 'scale_factor']
    features = [c for c in all_cols if c not in exclude_cols]
    
    categorical_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wday', 'month', 'snap_active', 'is_weekend', 'has_event']
    active_categoricals = [c for c in categorical_features if c in features]
    
    splits = get_cv_splits(N_FOLDS, END_TRAIN_DAY, FORECAST_HORIZON)
    fold_metrics = []

    # 4. Cross-Validation Loop
    for split in splits:
        print(f"\n   -> Slicing data for Fold {split['fold']}...")
        if TARGET_STORE == "ALL":
            train_df = lazy_df.filter((pl.col("d") <= split["train_end"])).collect().to_pandas()
            val_df = lazy_df.filter((pl.col("d") >= split["val_start"]) & (pl.col("d") <= split["val_end"])).collect().to_pandas()
        else:
            train_df = lazy_df.filter((pl.col("d") <= split["train_end"]) & (pl.col("store_id") == TARGET_STORE)).collect().to_pandas()
            val_df = lazy_df.filter((pl.col("d") >= split["val_start"]) & (pl.col("d") <= split["val_end"]) & (pl.col("store_id") == TARGET_STORE)).collect().to_pandas()

        # Failsafe for Weights
        if 'weight' not in train_df.columns: train_df['weight'] = 1.0; train_df['scale_factor'] = 1.0
        if 'weight' not in val_df.columns: val_df['weight'] = 1.0; val_df['scale_factor'] = 1.0

        y_val = val_df['sales'].to_numpy(dtype=np.float32)
        w_val = (val_df['weight'] / (val_df['scale_factor'] + 1e-5)).to_numpy(dtype=np.float32)
        val_identifiers = val_df[['item_id', 'd']].copy()

        # ==========================================================
        # AUTO-DETECTOR: Catch rogue text columns (like event_name)
        # ==========================================================
        string_cols = train_df[features].select_dtypes(include=['object', 'string']).columns.tolist()
        for col in string_cols:
            if col not in active_categoricals:
                active_categoricals.append(col)

        # THE FIX: Universal Categorical Synchronizer
        for col in active_categoricals:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype('category')
                val_df[col] = pd.Categorical(val_df[col], categories=train_df[col].cat.categories)

        print("   -> Executing custom model logic...")
        predictions = custom_model.run_model(train_df=train_df, val_df=val_df, val_identifiers=val_identifiers, features=features, categoricals=active_categoricals)

        rmse = root_mean_squared_error(y_val, predictions)
        mae = mean_absolute_error(y_val, predictions)
        wspl = wspl_metric(predictions, y_val, w_val, alpha=0.50)
        
        print(f"   [Fold {split['fold']} Results] WSPL: {wspl:.6f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        fold_metrics.append({'rmse': rmse, 'mae': mae, 'wspl': wspl})

        del train_df, val_df, predictions, y_val, w_val
        gc.collect()

    print(f"\n{'='*20} FINAL OOF AVERAGE {'='*20}")
    print(f"Global RMSE: {np.mean([f['rmse'] for f in fold_metrics]):.4f} | Global MAE: {np.mean([f['mae'] for f in fold_metrics]):.4f}")