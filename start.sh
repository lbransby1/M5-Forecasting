#!/bin/bash
# 1. Run the pipeline (Use the correct path!)
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh

# 2. Start the FastAPI backend
# Since the root is now /, we tell uvicorn where the 'app' object is
# Logic: [folder_name].[file_name]:[variable_name]
uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}