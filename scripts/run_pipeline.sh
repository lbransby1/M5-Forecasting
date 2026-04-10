#!/bin/bash
set -e

# 1. ENSURE WE ARE IN THE PROJECT ROOT
# This gets the directory where the script lives, then goes up one level to the root
# So no matter where you call this from, the paths 'backend/data' work.
cd "$(dirname "$0")/.."
echo "📍 Pipeline working directory set to: $(pwd)"

echo "🚀 [STAGE 1/3] Setting up environment..."
mkdir -p backend/data/raw
mkdir -p backend/data/processed
mkdir -p models/production_v1

# Sync requirements (using root requirements.txt)
pip install -r requirements.txt --quiet

echo "📥 [STAGE 2/3] Fetching latest M5 Data..."
python training/download_data.py --output_dir backend/data/raw

echo "🛠️ [STAGE 3/3] Running Polars Pre-processing (All Stores)..."
python training/pre_process.py \
    --input_dir backend/data/raw \
    --output_file backend/data/processed/m5_improved.parquet \
    --add_features

echo "✅ [SUCCESS] Backend data is synced and processed."