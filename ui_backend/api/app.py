from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ui_backend.api.routes import forecast, historical, simulate, insights
import os

app = FastAPI(title="AIgnition Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(historical.router, prefix="/api/historical", tags=["Historical"])
app.include_router(simulate.router, prefix="/api/simulate", tags=["Simulate"])
app.include_router(insights.router, prefix="/api/insights", tags=["Insights"])

@app.get("/api/health", tags=["Health"])
def health_check():
    from llm_Integration import config
    return {"status": "ok", "model": config.MODEL_NAME}


