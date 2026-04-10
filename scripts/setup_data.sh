#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 [STAGE 1/3] Setting up environment..."
mkdir -p backend/data/raw
mkdir -p backend/data/processed
mkdir -p models/production_v1

# Sync latest requirements
pip install -r requirements.txt --quiet

echo "📥 [STAGE 2/3] Fetching latest M5 Data..."
# Ensure your download_data.py is also set up to accept --output_dir
python training/download_data.py --output_dir backend/data/raw

echo "🛠️ [STAGE 3/3] Running Polars Pre-processing (All Stores)..."
# ADDED: --add_features flag so you get your rolling means!
python training/pre_process.py \
    --input_dir backend/data/raw \
    --output_file backend/data/processed/m5_improved.parquet \
    --add_features

echo "✅ [SUCCESS] Backend data is synced and processed."
echo "📍 Parquet Location: backend/data/processed/m5_improved.parquet"