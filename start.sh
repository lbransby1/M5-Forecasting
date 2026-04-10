#!/bin/bash

# 1. FAIL FAST: Exit if any command fails
set -e

echo "📍 Current Working Directory: $(pwd)"
echo "📂 Top-level files: $(ls -F)"

# 2. PATH DISCOVERY: Find the pipeline script
# We do this in case it's in /scripts or /training
PIPELINE_PATH=$(find . -name "run_pipeline.sh" | head -n 1)

if [ -z "$PIPELINE_PATH" ]; then
    echo "❌ FATAL: Could not find run_pipeline.sh in the repository!"
    exit 1
fi

echo "⚙️ Found MLOps Pipeline at: $PIPELINE_PATH"

# 3. FIX WINDOWS LINE ENDINGS (Important for Railway/Linux)
# If you edited this on Windows, it might have hidden characters that break scripts
sed -i 's/\r$//' "$PIPELINE_PATH"
chmod +x "$PIPELINE_PATH"

# 4. EXECUTE PIPELINE (Stage 1: Ingest & Preprocess)
echo "🚀 [STAGE 1/2] Running Data Pipeline..."
bash "$PIPELINE_PATH"

# 5. START PRODUCTION SERVER (Stage 2: API)
echo "🌐 [STAGE 2/2] Starting FastAPI Server..."
# We use 'backend.main' because the Python path starts at the root /
uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}