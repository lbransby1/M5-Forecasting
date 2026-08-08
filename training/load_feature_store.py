import hashlib
import os
import sys
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.core.redis_store import (
    REDIS_SERIES_DAYS,
    flush_namespace,
    get_client,
    set_meta,
    write_series,
)
from core.feature_engineering import HORIZON_BASE_FEATURES, REDIS_ROW_COLUMNS

DEFAULT_PARQUET = BASE_DIR / "backend" / "data" / "processed" / "m5_improved.parquet"
RAW_DIR = BASE_DIR / "backend" / "data" / "raw"


def compute_dataset_version(parquet_path: Path) -> str:
    parts = []
    if parquet_path.exists():
        stat = parquet_path.stat()
        parts.append(f"parquet:{stat.st_mtime_ns}:{stat.st_size}")

    for name in ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]:
        raw_path = RAW_DIR / name
        if raw_path.exists():
            stat = raw_path.stat()
            parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def load_feature_store(parquet_path: Path = DEFAULT_PARQUET, force: bool = False) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")

    version = compute_dataset_version(parquet_path)
    client = get_client()
    existing = client.get("m5:meta:version")
    if existing == version and not force:
        print(f"[skip] Redis feature store already at version {version}")
        return

    print(f"[load] Building Redis feature store from {parquet_path}")
    columns = list(dict.fromkeys(["store_id", "item_id"] + REDIS_ROW_COLUMNS))
    parquet_cols = pl.read_parquet(parquet_path, n_rows=0).columns
    missing = [c for c in columns if c not in parquet_cols]
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}. Re-run core/pre_process.py.")

    lf = (
        pl.scan_parquet(parquet_path)
        .select(columns)
        .sort(["store_id", "item_id", "d"])
    )
    df = lf.collect()

    flush_namespace()
    pipe = client.pipeline(transaction=False)
    batch = 0

    for (store_id, item_id), group in df.group_by(["store_id", "item_id"], maintain_order=True):
        rows = group.tail(REDIS_SERIES_DAYS).to_dicts()
        write_series(str(store_id), str(item_id), rows, pipe)
        batch += 1
        if batch % 500 == 0:
            pipe.execute()
            pipe = client.pipeline(transaction=False)
            print(f"   -> loaded {batch} item-store series")

    max_d = int(df["d"].max())
    set_meta(version, max_d, pipe)
    pipe.execute()
    print(f"[ok] Redis feature store loaded: version={version}, max_d={max_d}, series={batch}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(DEFAULT_PARQUET))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_feature_store(Path(args.parquet), force=args.force)
