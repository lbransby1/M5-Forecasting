import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import time

# Configuration from notebook
QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
SEQ_LENGTH = 28
SAVE_PATH = "models/"
os.makedirs(SAVE_PATH, exist_ok=True)

def build_foods_training_set():
    print("🚀 Downloading and filtering M5 data for FOODS items in CA_3...")
    # Loading directly from your notebook's sources
    sales = pd.read_csv("https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv")
    
    # Filter for all Food items in CA_3
    # M5 food items are categorized under 'FOODS_1', 'FOODS_2', and 'FOODS_3'
    df_foods = sales[(sales['store_id'] == 'CA_3') & (sales['cat_id'] == 'FOODS')].copy()
    print(f"✅ Found {len(df_foods)} food items in store CA_3.")
    
    day_cols = [c for c in df_foods.columns if c.startswith('d_')]
    data_array = df_foods[day_cols].values.astype(np.float32)
    
    X, y = [], []
    # Sliding window logic for LightGBM training (as shown in notebook concepts)
    for i in range(data_array.shape[0]):
        for t in range(0, data_array.shape[1] - SEQ_LENGTH - 1, 7): # Stride 7 to keep training set size manageable
            window = data_array[i, t : t + SEQ_LENGTH]
            target = data_array[i, t + SEQ_LENGTH]
            X.append(window)
            y.append(target)
            
    # Save metadata for front-end selectors (Item ID, Dept)
    metadata = df_foods[['item_id', 'dept_id', 'cat_id']].drop_duplicates()
    metadata.to_csv("data/food_item_metadata.csv", index=False)
    
    return np.array(X), np.array(y)

def train_quantile_models():
    X, y = build_foods_training_set()
    lgb_train = lgb.Dataset(X, y)
    
    print(f"\n🚀 Starting Training for {len(QUANTILES)} Quantile Models...")
    start_total = time.time()
    
    for i, q in enumerate(QUANTILES):
        start_q = time.time()
        print(f"[{i+1}/{len(QUANTILES)}] Training Quantile: {q}...", end=" ")
        
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbosity': -1,
            'n_jobs': -1 # Use all cores for speed
        }
        
        # Train and save progress
        model = lgb.train(params, lgb_train, num_boost_round=100)
        joblib.dump(model, f"{SAVE_PATH}lgbm_q_{q}.pkl")
        
        elapsed_q = time.time() - start_q
        print(f"Done! ({elapsed_q:.2f}s)")
    
    total_time = time.time() - start_total
    print(f"\n✅ All 9 models saved to /models in {total_time:.2f}s.")

if __name__ == "__main__":
    train_quantile_models()