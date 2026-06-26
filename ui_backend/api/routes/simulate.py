import os
import json
from fastapi import APIRouter, HTTPException
from ui_backend.api.schemas import BudgetSimulationRequest, SimulationResponse

router = APIRouter()
FORECAST_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Forecasting', 'forecast_output.json'))

@router.post("", response_model=SimulationResponse)
def simulate_budget(request: BudgetSimulationRequest):
    # Load base forecast
    if not os.path.exists(FORECAST_FILE):
        raise HTTPException(status_code=404, detail="Base forecast not found. Please run the forecasting pipeline first.")
        
    with open(FORECAST_FILE, 'r') as f:
        base_data = json.load(f)
        
    base_budget = base_data.get("total_budget_input", 1)
    new_budget = request.total_budget if request.total_budget else base_budget
    
    # Simple allocation based on request channel split
    if request.channel_split:
        splits = request.channel_split
    else:
        # Default split matches existing
        splits = {}
        for ch, ch_data in base_data.get("channels", {}).items():
            splits[ch] = ch_data.get("budget_allocated", 0) / base_budget
            
    simulated_channels = {}
    total_rev_p10, total_rev_p50, total_rev_p90 = 0, 0, 0
    
    # Use standard elasticity of 0.75 (from budget_simulator.py)
    elasticity = 0.75
    
    for ch, ch_data in base_data.get("channels", {}).items():
        base_ch_budget = ch_data.get("budget_allocated", 0)
        
        # New budget for this channel
        split_pct = splits.get(ch, 0)
        new_ch_budget = new_budget * split_pct
        
        # Scale revenue using elasticity: R_new = R_old * (B_new / B_old)^0.75
        multiplier = (new_ch_budget / base_ch_budget)**elasticity if base_ch_budget > 0 else 0
        
        new_p10 = ch_data.get("revenue", {}).get("P10", 0) * multiplier
        new_p50 = ch_data.get("revenue", {}).get("P50", 0) * multiplier
        new_p90 = ch_data.get("revenue", {}).get("P90", 0) * multiplier
        
        simulated_channels[ch] = {
            "budget_allocated": round(new_ch_budget, 2),
            "revenue": {
                "P10": round(new_p10, 2),
                "P50": round(new_p50, 2),
                "P90": round(new_p90, 2)
            },
            "roas": {
                "P10": round(new_p10 / new_ch_budget, 2) if new_ch_budget > 0 else 0,
                "P50": round(new_p50 / new_ch_budget, 2) if new_ch_budget > 0 else 0,
                "P90": round(new_p90 / new_ch_budget, 2) if new_ch_budget > 0 else 0
            }
        }
        
        total_rev_p10 += new_p10
        total_rev_p50 += new_p50
        total_rev_p90 += new_p90

    return {
        "scenario": request.scenario,
        "total_budget": round(new_budget, 2),
        "total_revenue": {
            "P10": round(total_rev_p10, 2),
            "P50": round(total_rev_p50, 2),
            "P90": round(total_rev_p90, 2)
        },
        "blended_roas": {
            "P10": round(total_rev_p10 / new_budget, 2) if new_budget > 0 else 0,
            "P50": round(total_rev_p50 / new_budget, 2) if new_budget > 0 else 0,
            "P90": round(total_rev_p90 / new_budget, 2) if new_budget > 0 else 0
        },
        "channels": simulated_channels
    }
