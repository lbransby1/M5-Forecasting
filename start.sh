#!/bin/bash

LOCKFILE="/tmp/myscript.lock"

# Check if the lock file exists
if [ -f "$LOCKFILE" ]; then
    echo "Script is already running. Exiting."
    exit 0
fi

# Create the lock file
touch "$LOCKFILE"

# Ensure the lock file is removed when the script finishes (or crashes)
trap 'rm -f "$LOCKFILE"' EXIT

# --- YOUR EXISTING STARTUP CODE BELOW ---
echo "Starting application..."

set -e

echo "Current Working Directory: $(pwd)"


# 1. Manually check the two most likely locations
if [ -f "./scripts/run_pipeline.sh" ]; then
    PIPELINE_PATH="./scripts/run_pipeline.sh"

else
    echo "FATAL: run_pipeline.sh not found in /scripts or /training"
    echo "Checking scripts folder content: $(ls -F scripts/ 2>/dev/null || echo 'Folder not found')"
    echo "Checking training folder content: $(ls -F training/ 2>/dev/null || echo 'Folder not found')"
    exit 1
fi

echo "⚙️ Found MLOps Pipeline at: $PIPELINE_PATH"

# 2. Fix Windows line endings and set permissions
sed -i 's/\r$//' "$PIPELINE_PATH"
chmod +x "$PIPELINE_PATH"

# 3. Run Pipeline
echo "[STAGE 1/2] Running Data Pipeline..."
bash "$PIPELINE_PATH"

# 4. Start API
echo "[STAGE 2/2] Starting FastAPI Server..."
# The ${PORT:-8000} syntax means: Use $PORT if it exists, otherwise use 8000
uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}