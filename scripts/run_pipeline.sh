#!/bin/bash
set -e

cd "$(dirname "$0")/.."

PROCESSED_FILE="backend/data/processed/m5_improved.parquet"
RAW_DIR=""

resolve_raw_dir() {
  python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from core.pre_process import resolve_raw_dir
print(resolve_raw_dir(os.environ.get("RAW_DIR_OVERRIDE") or None))
PY
}

RAW_DIR=""
if RAW_DIR="$(resolve_raw_dir 2>/dev/null)"; then
  echo "Found raw data at: $RAW_DIR"
else
  RAW_DIR="backend/data/raw"
  echo "Raw data not found; will download to $RAW_DIR"
fi

mkdir -p backend/data/raw
mkdir -p backend/data/processed

compute_version() {
  python - <<'PY'
import hashlib
from pathlib import Path

parts = []
processed = Path("backend/data/processed/m5_improved.parquet")
raw_dir = Path("backend/data/raw")
if processed.exists():
    stat = processed.stat()
    parts.append(f"parquet:{stat.st_mtime_ns}:{stat.st_size}")
for name in ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]:
    raw = Path("backend/data/raw") / name
    if not raw.exists():
        raw = Path("data/raw") / name
    if raw.exists():
        stat = raw.stat()
        parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
print(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16])
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

if [ -n "${REDIS_URL:-}" ] && [ "$REDIS_VERSION" = "$TARGET_VERSION" ] && [ -f "$PROCESSED_FILE" ]; then
  echo "[SKIP] Redis feature store already matches dataset version $TARGET_VERSION."
elif [ -f "$PROCESSED_FILE" ] && [ ! -f "$RAW_DIR/sales_train_evaluation.csv" ]; then
  echo "[SKIP] Processed parquet exists; raw files absent."
else
  if [ ! -f "$PROCESSED_FILE" ] || [ -f "$RAW_DIR/sales_train_evaluation.csv" ]; then
    echo "Fetching latest M5 Data..."
    python core/download_data.py --output_dir backend/data/raw

    echo "Running local preprocessing..."
    RAW_DIR="$(resolve_raw_dir 2>/dev/null || echo backend/data/raw)"
    python core/pre_process.py \
      --mode local \
      --raw_dir "$RAW_DIR" \
      --output_path "$PROCESSED_FILE"
  fi
fi

if [ -n "${REDIS_URL:-}" ]; then
  echo "Loading Redis feature store..."
  python training/load_feature_store.py --parquet "$PROCESSED_FILE"
fi

echo "Ready for API startup."
