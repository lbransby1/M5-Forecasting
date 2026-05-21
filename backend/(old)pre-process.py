import pandas as pd
import numpy as np
import os

# 1. Configuration
STORE_ID = 'CA_3'
CAT_ID = 'FOODS'
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def create_popular_processed_data():
    print(f"🚀 Downloading M5 Data for {CAT_ID} in {STORE_ID}...")
    url = "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv"
    sales = pd.read_csv(url)
    
    # Filter scope
    df = sales[(sales['store_id'] == STORE_ID) & (sales['cat_id'] == CAT_ID)].copy()
    day_cols = [c for c in df.columns if c.startswith('d_')]
    
    # --- 1. CALCULATE LEADERBOARD METRICS ---
    print("📈 Calculating metrics...")
    df['Popularity'] = df[day_cols].sum(axis=1)
    
    # Use last 28 days for Volatility
    recent_28 = day_cols[-28:]
    df['Volatility'] = (df[recent_28].std(axis=1) / df[recent_28].mean(axis=1)).fillna(0).round(2)
    
    # Use last 100 days for Sparsity
    df['Sparsity'] = ((df[day_cols[-100:]] == 0).sum(axis=1) / 100 * 100).round(1)

    df = df.sort_values(by='Popularity', ascending=False)

    # --- 2. SAVE PROCESSED SALES ---
    recent_days = day_cols[-100:]
    df[['item_id', 'dept_id'] + recent_days].to_csv(f"{DATA_DIR}/processed_sales.csv", index=False)
    
    # --- 3. SAVE ITEM LEADERBOARD (The critical part) ---
    leaderboard_cols = ['item_id', 'dept_id', 'Popularity', 'Volatility', 'Sparsity']
    leaderboard = df[leaderboard_cols].copy()
    
    map_path = f"{DATA_DIR}/product_mapping.csv"
    if os.path.exists(map_path):
        names = pd.read_csv(map_path)
        
        # FIX: Drop dept_id from 'names' if it exists there to prevent dept_id_x / dept_id_y
        if 'dept_id' in names.columns and 'dept_id' in leaderboard.columns:
            names = names.drop(columns=['dept_id'])
            
        leaderboard = leaderboard.merge(names, on="item_id", how="left")
    
    # Ensure product_name exists
    if 'product_name' not in leaderboard.columns:
        leaderboard['product_name'] = leaderboard['item_id']
    else:
        leaderboard['product_name'] = leaderboard['product_name'].fillna(leaderboard['item_id'])
        
    leaderboard.to_csv(f"{DATA_DIR}/item_leaderboard.csv", index=False)
    
    # --- 4. SAVE METADATA ---
    df[['item_id', 'dept_id', 'cat_id']].to_csv(f"{DATA_DIR}/food_item_metadata.csv", index=False)
    
    print(f"✅ Success! Generated all files. Top Item: {df.iloc[0]['item_id']}")

if __name__ == "__main__":
    create_popular_processed_data()