import os
import json
from typing import Any, Optional

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_SERIES_DAYS = int(os.environ.get("REDIS_SERIES_DAYS", "112"))
REDIS_HISTORY_DAYS = int(os.environ.get("REDIS_HISTORY_DAYS", "84"))

_client: Optional[redis.Redis] = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def init_redis() -> redis.Redis:
    client = get_client()
    client.ping()
    return client


def meta_version() -> Optional[str]:
    return get_client().get("m5:meta:version")


def meta_max_d() -> Optional[int]:
    value = get_client().get("m5:meta:max_d")
    return int(value) if value is not None else None


def ctx_key(store_id: str, item_id: str) -> str:
    return f"m5:ctx:{store_id}:{item_id}"


def row_key(store_id: str, item_id: str, d: int) -> str:
    return f"m5:row:{store_id}:{item_id}:{d}"


def sales_key(store_id: str, item_id: str) -> str:
    return f"m5:sales:{store_id}:{item_id}"


def _cast_value(key: str, value: str) -> Any:
    if key in {"d", "wday", "month", "horizon_day", "target_wday", "days_since_last_sale_lag_28"}:
        return int(float(value))
    if key in {"item_id", "dept_id", "cat_id", "store_id", "state_id"}:
        return value
    if key in {"snap_CA", "snap_TX", "snap_WI"}:
        return int(float(value))
    return float(value)


def hash_to_dict(data: dict[str, str]) -> Optional[dict[str, Any]]:
    if not data:
        return None
    return {k: _cast_value(k, v) for k, v in data.items()}


def get_ctx(store_id: str, item_id: str) -> Optional[dict[str, Any]]:
    return hash_to_dict(get_client().hgetall(ctx_key(store_id, item_id)))


def get_row_at_d(store_id: str, item_id: str, d: int) -> Optional[dict[str, Any]]:
    return hash_to_dict(get_client().hgetall(row_key(store_id, item_id, d)))


def get_sales_history(store_id: str, item_id: str, count: int = REDIS_HISTORY_DAYS) -> list[float]:
    values = get_client().lrange(sales_key(store_id, item_id), -count, -1)
    return [float(v) for v in values] if values else [0.0] * count


def _serialize_value(key: str, value) -> str:
    if value is None:
        return "0"
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return "0"
    return str(value)


def write_row(store_id: str, item_id: str, row: dict[str, Any], pipe: redis.client.Pipeline) -> None:
    payload = {k: _serialize_value(k, row[k]) for k in row.keys()}
    d = int(row["d"])
    pipe.hset(row_key(store_id, item_id, d), mapping=payload)


def write_series(store_id: str, item_id: str, rows: list[dict[str, Any]], pipe: redis.client.Pipeline) -> None:
    if not rows:
        return

    latest = rows[-1]
    pipe.hset(ctx_key(store_id, item_id), mapping={k: _serialize_value(k, latest[k]) for k in latest.keys()})

    sales_values = [str(r["sales"]) for r in rows[-REDIS_HISTORY_DAYS:]]
    s_key = sales_key(store_id, item_id)
    pipe.delete(s_key)
    if sales_values:
        pipe.rpush(s_key, *sales_values)

    for row in rows:
        write_row(store_id, item_id, row, pipe)


def set_meta(version: str, max_d: int, pipe: Optional[redis.client.Pipeline] = None) -> None:
    client = pipe or get_client()
    client.set("m5:meta:version", version)
    client.set("m5:meta:max_d", str(max_d))


def flush_namespace() -> None:
    client = get_client()
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match="m5:*", count=5000)
        if keys:
            pipe = client.pipeline(transaction=False)
            for key in keys:
                pipe.unlink(key)
            pipe.execute()
            deleted += len(keys)
            if deleted % 50000 == 0 or cursor == 0:
                print(f"      flushed {deleted:,} keys...")
        if cursor == 0:
            break
    if deleted:
        print(f"      flushed {deleted:,} keys total")
