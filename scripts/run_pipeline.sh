#!/bin/bash
set -e

# Move to project root
cd "$(dirname "$0")/.."

# Define the final target file
PROCESSED_FILE="backend/data/processed/m5_improved.parquet"

echo "Setting up environment..."
mkdir -p backend/data/raw
mkdir -p backend/data/processed

# CHECK: Does the final data already exist?
if [ -f "$PROCESSED_FILE" ]; then
    echo "[SKIP] Processed data already exists at $PROCESSED_FILE."
    echo "Skipping Download and Preprocessing stages."
else
    echo "Fetching latest M5 Data..."
    python training/download_data.py --output_dir backend/data/raw

    echo "Uploading M5 Walmart Data to S3..."
    python training/upload_data.py

    echo "Running Polars Pre-processing (All Stores)..."
    python training/pre_process.py \
        --input_dir backend/data/raw \
        --output_file "$PROCESSED_FILE" \
        --add_features

    echo "SUCCESS: Backend data is synced and processed."
fi

echo "Ready for API startup."