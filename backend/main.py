# backend/main.py
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas.payloads import ForecastResponse, LeaderboardResponse
from core.data import load_data_assets, fetch_leaderboard
from core.inference import load_ml_models, generate_full_forecast

app = FastAPI(title="M5 Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_data_assets()
    load_ml_models()

@app.get("/predict/{item_id}", response_model=ForecastResponse)
def predict(item_id: str, store_id: str):
    try:
        results = generate_full_forecast(item_id, store_id)
        return ForecastResponse(**results)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard():
    data = fetch_leaderboard()
    return LeaderboardResponse(data=data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)