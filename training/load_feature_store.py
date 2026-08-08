import hashlib
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
from core.feature_engineering import REDIS_ROW_COLUMNS

DEFAULT_PARQUET = BASE_DIR / "backend" / "data" / "processed" / "m5_improved.parquet"
RAW_DIR = BASE_DIR / "backend" / "data" / "raw"
CAT_STRING_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]


def resolve_parquet_uri() -> str:
    uri = os.environ.get("PROCESSED_PARQUET_URI", "").strip()
    if uri:
        return uri
    return str(DEFAULT_PARQUET)


def s3_storage_options() -> dict[str, Any]:
    endpoint = os.getenv("STORAGE_ENDPOINT_URL")
    if not endpoint:
        return {}
    return {
        "endpoint_url": endpoint,
        "key": os.getenv("STORAGE_ACCESS_KEY_ID"),
        "secret": os.getenv("STORAGE_SECRET_ACCESS_KEY"),
    }


def _parquet_kwargs(parquet_ref: str) -> dict[str, Any]:
    if parquet_ref.startswith("s3://"):
        opts = s3_storage_options()
        if not opts:
            raise ValueError(
                "PROCESSED_PARQUET_URI is s3:// but STORAGE_ENDPOINT_URL / credentials are not set."
            )
        return {"storage_options": opts}
    return {}


def parquet_exists(parquet_ref: str) -> bool:
    if parquet_ref.startswith("s3://"):
        try:
            import boto3

            parsed = urlparse(parquet_ref)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            client = boto3.client(
                "s3",
                endpoint_url=os.getenv("STORAGE_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("STORAGE_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("STORAGE_SECRET_ACCESS_KEY"),
                region_name=os.getenv("STORAGE_REGION", "auto"),
            )
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return Path(parquet_ref).exists()


def compute_dataset_version(parquet_ref: str | None = None) -> str:
    override = os.environ.get("DATASET_VERSION", "").strip()
    if override:
        return override

    parquet_ref = parquet_ref or resolve_parquet_uri()

    if parquet_ref.startswith("s3://"):
        try:
            import boto3

            parsed = urlparse(parquet_ref)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            client = boto3.client(
                "s3",
                endpoint_url=os.getenv("STORAGE_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("STORAGE_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("STORAGE_SECRET_ACCESS_KEY"),
                region_name=os.getenv("STORAGE_REGION", "auto"),
            )
            head = client.head_object(Bucket=bucket, Key=key)
            etag = head.get("ETag", "").strip('"')
            size = head.get("ContentLength", 0)
            digest = hashlib.sha256(f"s3:{bucket}/{key}:{etag}:{size}".encode()).hexdigest()[:16]
            return digest
        except Exception as exc:
            print(f"[warn] Could not compute S3 dataset version: {exc}")
            return hashlib.sha256(parquet_ref.encode()).hexdigest()[:16]

    parts = []
    path = Path(parquet_ref)
    if path.exists():
        stat = path.stat()
        parts.append(f"parquet:{stat.st_mtime_ns}:{stat.st_size}")

    for name in ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv"]:
        for raw_path in (RAW_DIR / name, BASE_DIR / "data" / "raw" / name):
            if raw_path.exists():
                stat = raw_path.stat()
                parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
                break

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def load_feature_store(parquet_ref: str | None = None, force: bool = False) -> None:
    parquet_ref = parquet_ref or resolve_parquet_uri()
    if not parquet_exists(parquet_ref):
        raise FileNotFoundError(f"Processed parquet not found: {parquet_ref}")

    parquet_kwargs = _parquet_kwargs(parquet_ref)
    version = compute_dataset_version(parquet_ref)
    client = get_client()
    existing = client.get("m5:meta:version")
    if existing == version and not force:
        print(f"[skip] Redis feature store already at version {version}")
        return

    print(f"[load] Building Redis feature store from {parquet_ref}")
    columns = list(dict.fromkeys(["store_id", "item_id"] + REDIS_ROW_COLUMNS))
    parquet_cols = pl.read_parquet(parquet_ref, n_rows=0, **parquet_kwargs).columns
    missing = [c for c in columns if c not in parquet_cols]
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}. Re-run core/pre_process.py.")

    print("[load] Flushing old Redis keys (remote Redis can take several minutes)...")
    flush_namespace()
    pipe = client.pipeline(transaction=False)
    batch = 0
    max_d = 0

    for store in STORES:
        print(f"   -> store {store}")
        df = (
            pl.scan_parquet(parquet_ref, **parquet_kwargs)
            .filter(pl.col("store_id") == store)
            .select(columns)
            .sort(["store_id", "item_id", "d"])
            .collect()
        )
        if df.is_empty():
            continue

        cat_cols = [c for c in CAT_STRING_COLS if c in df.columns]
        if cat_cols:
            df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in cat_cols])

        max_d = max(max_d, int(df["d"].max()))

        for (store_id, item_id), group in df.group_by(["store_id", "item_id"], maintain_order=True):
            rows = group.tail(REDIS_SERIES_DAYS).to_dicts()
            write_series(str(store_id), str(item_id), rows, pipe)
            batch += 1
            if batch % 500 == 0:
                pipe.execute()
                pipe = client.pipeline(transaction=False)
                print(f"      loaded {batch} item-store series")

        del df

    set_meta(version, max_d, pipe)
    pipe.execute()
    print(f"[ok] Redis feature store loaded: version={version}, max_d={max_d}, series={batch}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=None, help="Local path or s3:// URI")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_feature_store(args.parquet or resolve_parquet_uri(), force=args.force)
