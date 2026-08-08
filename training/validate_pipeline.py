"""Validate horizon v2 + Redis feature store integration."""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

REQUIRED_COLUMNS = [
    "lag_28", "roll_mean_7_lag_28", "roll_mean_28_lag_28",
    "price_momentum_7d", "price_momentum_28d",
    "masked_roll_mean_28_lag_28", "ema_lag_28", "days_since_last_sale_lag_28",
]


def main():
    parquet = BASE_DIR / "backend/data/processed/m5_improved.parquet"
    if not parquet.exists():
        print("[fail] Missing processed parquet. Run: python core/pre_process.py --mode local")
        return 1

    import polars as pl
    cols = pl.read_parquet(parquet, n_rows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        print(f"[fail] Parquet missing horizon columns: {missing}")
        print("Re-run preprocessing with updated feature engineering.")
        return 1
    print("[ok] Parquet has horizon feature columns")

    if os.environ.get("FEATURE_STORE") == "redis" and os.environ.get("REDIS_URL"):
        try:
            import redis
            from training.load_feature_store import load_feature_store
            from backend.core.redis_store import get_client, get_ctx

            load_feature_store(parquet, force=False)
            client = get_client()
            client.ping()
            version = client.get("m5:meta:version")
            ctx_count = len(list(client.scan_iter("m5:ctx:*", count=1000)))
            print(f"[ok] Redis loaded: version={version}, ctx_keys={ctx_count}")

            sample = get_ctx("CA_1", "FOODS_3_684")
            if sample and "lag_28" in sample:
                print("[ok] Sample Redis ctx has lag_28")
            else:
                print("[warn] Sample ctx missing or incomplete")
        except Exception as exc:
            print(f"[skip] Redis check failed ({exc}). Use FEATURE_STORE=parquet locally.")
    elif os.environ.get("REDIS_URL") and os.environ.get("FEATURE_STORE") != "redis":
        print("[skip] REDIS_URL set but FEATURE_STORE is not redis")
    else:
        print("[skip] Redis not configured (FEATURE_STORE=parquet)")

    model_dir = BASE_DIR / "models/model_horizon_v2"
    models = list(model_dir.glob("global_model_alpha_*.txt")) if model_dir.exists() else []
    if len(models) == 9:
        print("[ok] Found 9 horizon v2 model files")
    else:
        print(f"[warn] Found {len(models)}/9 v2 models. Run: python training/train_model_horizon_v2.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
