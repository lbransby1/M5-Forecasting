#!/bin/bash
set -e

cd "$(dirname "$0")/.."

PROCESSED_FILE="backend/data/processed/m5_improved.parquet"
PARQUET_URI="${PROCESSED_PARQUET_URI:-}"

mkdir -p backend/data/raw
mkdir -p backend/data/processed

compute_version() {
  python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from training.load_feature_store import compute_dataset_version, resolve_parquet_uri

print(compute_dataset_version(resolve_parquet_uri()))
PY
}

TARGET_VERSION="$(compute_version)"
REDIS_VERSION=""
if [ -n "${REDIS_URL:-}" ]; then
  REDIS_VERSION="$(python - <<'PY'
import os
try:
    import redis
    client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    print(client.get("m5:meta:version") or "")
except Exception:
    print("")
PY
)"
fi

REDIS_PARQUET="$(python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from training.load_feature_store import resolve_parquet_uri
print(resolve_parquet_uri())
PY
)"

if [ "${SKIP_DATA_PIPELINE:-}" = "true" ]; then
  echo "[SKIP] SKIP_DATA_PIPELINE=true — no download/preprocess on this server."
elif [ -n "$PARQUET_URI" ]; then
  echo "[SKIP] PROCESSED_PARQUET_URI set — using remote processed data: $PARQUET_URI"
elif [ -f "$PROCESSED_FILE" ]; then
  echo "[SKIP] Local processed parquet already exists."
else
  echo "Fetching latest M5 Data..."
  python core/download_data.py --output_dir backend/data/raw

  echo "Running local preprocessing..."
  python core/pre_process.py \
    --mode local \
    --raw_dir backend/data/raw \
    --output_path "$PROCESSED_FILE"
fi

if [ "${SKIP_REDIS_LOAD:-}" = "true" ]; then
  echo "[SKIP] SKIP_REDIS_LOAD=true"
elif [ -n "${REDIS_URL:-}" ] && [ -n "$REDIS_VERSION" ] && [ "$REDIS_VERSION" = "$TARGET_VERSION" ]; then
  echo "[SKIP] Redis feature store already matches dataset version $TARGET_VERSION."
else
  if [ -n "${REDIS_URL:-}" ]; then
    echo "Loading Redis feature store from $REDIS_PARQUET ..."
    python training/load_feature_store.py --parquet "$REDIS_PARQUET"
  else
    echo "[SKIP] REDIS_URL not set — skipping Redis load."
  fi
fi

echo "Ready for API startup."
