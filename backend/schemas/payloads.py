# backend/schemas/payloads.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    product_name: str
    history: List[float]
    backtest: Dict[str, List[float]]
    forecast: Dict[str, List[float]]
    metrics: Dict[str, float]

class LeaderboardResponse(BaseModel):
    data: List[Dict[str, Any]]