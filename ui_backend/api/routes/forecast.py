import os
import json
import subprocess
from fastapi import APIRouter, HTTPException, Query
from ui_backend.api.schemas import ForecastOverviewResponse, ChannelBreakdownResponse

router = APIRouter()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

FORECAST_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Forecasting', 'forecast_output.json'))

def load_forecast():
    if not os.path.exists(FORECAST_FILE):
        raise HTTPException(status_code=404, detail="Forecast output not found. Please run the forecasting pipeline first.")
    with open(FORECAST_FILE, 'r') as f:
        return json.load(f)

@router.get("", response_model=ForecastOverviewResponse)
def get_forecast_overview():
    data = load_forecast()
    return {
        "forecast_period_days": data.get("forecast_period_days", 30),
        "scenario": data.get("scenario", "Baseline"),
        "total_budget": data.get("total_budget_input", 0.0),
        "revenue": data.get("revenue", {"P10": 0, "P50": 0, "P90": 0}),
        "blended_roas": data.get("blended_roas", {"P10": 0, "P50": 0, "P90": 0})
    }

@router.get("/channels", response_model=ChannelBreakdownResponse)
def get_channel_breakdown():
    data = load_forecast()
    return {"channels": data.get("channels", {})}

@router.post("/generate")
def generate_forecast(days: int = Query(30, description="Forecast window in days (30, 60, or 90)")):
    if days not in [30, 60, 90]:
        raise HTTPException(status_code=400, detail="Days must be 30, 60, or 90.")
        
    try:
        # Run the Forecasting script as a subprocess from the Forecasting directory
        script_path = os.path.join(PROJECT_ROOT, 'Forecasting', 'main.py')
        forecasting_dir = os.path.join(PROJECT_ROOT, 'Forecasting')
        result = subprocess.run(
            ["python", "main.py", "--days", str(days)],
            cwd=forecasting_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Forecasting script failed: {result.stderr}")
            
        return {"status": "success", "message": f"Forecast generated for {days} days."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
