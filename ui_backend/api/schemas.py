from pydantic import BaseModel
from typing import Dict, Optional, Any, List

# --- /api/simulate Schemas ---

class BudgetSimulationRequest(BaseModel):
    scenario: str = "custom"
    total_budget: Optional[float] = None
    channel_split: Optional[Dict[str, float]] = None

class ConfidenceInterval(BaseModel):
    P10: float
    P50: float
    P90: float
    currency: Optional[str] = None

class SimulationResponse(BaseModel):
    scenario: str
    total_budget: float
    total_revenue: ConfidenceInterval
    blended_roas: ConfidenceInterval
    channels: Dict[str, Any]

# --- /api/forecast Schemas ---

class ForecastOverviewResponse(BaseModel):
    forecast_period_days: int
    scenario: str
    total_budget: float
    revenue: ConfidenceInterval
    blended_roas: ConfidenceInterval

class ChannelBreakdownResponse(BaseModel):
    channels: Dict[str, Any]

# --- /api/insights Schemas ---

class InsightsResponse(BaseModel):
    summary: str
    key_drivers: List[str]
    risks: List[str]
    opportunities: List[str]
    anomalies: List[str]
    confidence_level: str
    metadata: Dict[str, Any]
