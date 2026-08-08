import lightgbm as lgb
import numpy as np
import tabulate
import pandas as pd

import pandas as pd
import os

def generate_kaggle_submission(val_df, predictions, raw_dir="data/raw"):
    """
    Merges local predictions perfectly into Kaggle's official sample_submission.csv
    This guarantees 100% acceptance by the Kaggle grader.
    """
    print("\nFormatting predictions for Kaggle...")
    
    # 1. Rebuild the EXACT Kaggle ID string robustly
    results = val_df[['item_id', 'store_id', 'd']].copy()
    results['id'] = results['item_id'].astype(str) + '_' + results['store_id'].astype(str) + '_validation'
    results['sales'] = predictions

    # 2. Dynamically map unique validation days to F1-F28
    unique_days = sorted(results['d'].unique())
    day_mapping = {day: f"F{i+1}" for i, day in enumerate(unique_days)}
    results['F_day'] = results['d'].map(day_mapping)
    
    # 3. Pivot our local predictions to wide format
    my_sub = results.pivot(index='id', columns='F_day', values='sales').reset_index()
    
    # 4. Load the official blank Kaggle template
    # (Ensure this file is downloaded from Kaggle and sitting in your data/raw folder)
    sample_path = os.path.join(raw_dir, "accuracy_sample_submission.csv")
    if not os.path.exists(sample_path):
        print(f"ERROR: Could not find {sample_path}. Please download it from the Kaggle Data page and place it in {raw_dir}.")
        return
        
    print("📥 Loading official Kaggle sample_submission.csv...")
    official_sub = pd.read_csv(sample_path)
    
    # 5. The Injector: Update the official blank template with our predictions
    official_sub = official_sub.set_index('id')
    my_sub = my_sub.set_index('id')
    
    # This safely overwrites ONLY the rows we predicted (e.g., CA_1). 
    # The rest (TX, WI, and _evaluation) safely remain exactly as Kaggle expects them.
    official_sub = official_sub.astype(float)
    my_sub = my_sub.astype(float)
    official_sub.update(my_sub)
    
    # 6. Save and export
    final_submission = official_sub.reset_index()
    output_file = "submission.csv"
    final_submission.to_csv(output_file, index=False)
    
    print(f"✅ Bulletproof submission saved to {output_file}. Rows: {len(final_submission)}/60980")
    return final_submission

def run_model(train_df, val_df, val_identifiers, features, categoricals):
    # 1. Define Features, Targets, & Weights
    X_train = train_df[features].copy()
    y_train = train_df['sales'].values
    w_train = np.sqrt(train_df['weight'] / (train_df['scale_factor'] + 1)).values
    
    X_val = val_df[features].copy()
    y_val = val_df['sales'].values
    w_val = np.sqrt(val_df['weight'] / (val_df['scale_factor'] + 1)).values

    # 2. Create LightGBM Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 3. The Corrected Tweedie Parameters
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.15, 
        'metric': 'rmse',              
        'learning_rate': 0.03,         # Slower, more careful learning
        'num_leaves': 255,             # Massive increase in tree capacity
        'min_data_in_leaf': 100,       
        'verbose': -1,
        'random_state': 42
    }

    # Pick one specific item to trace through time
    sample_item = train_df[train_df['item_id'] == train_df['item_id'].iloc[0]].sort_values('d')

    # Print the target, the past, and the present/future side-by-side
    cols_to_check = ['d', 'sales', 'lag_1', 'roll_mean_7', 'is_weekend', 'sell_price']
    print(sample_item[cols_to_check].head(5).to_markdown())

    print("      -> Training Tweedie Model...")   
    model = lgb.train(
        params, 
        train_data, 
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=2000, 
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # More patience
            lgb.log_evaluation(period=50)            # Print every 50 rounds so you can watch it
        ]
    )

    # 4. Predict
    predictions = model.predict(X_val)


    generate_kaggle_submission(val_df, predictions)

    return predictions.astype(np.float32)