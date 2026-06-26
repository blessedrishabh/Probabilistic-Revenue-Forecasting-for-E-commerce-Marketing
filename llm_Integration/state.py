from typing import TypedDict, List, Dict, Any

class CausalAnalysisState(TypedDict):
    # ─── Inputs (set by load_context) ───
    forecast_data: Dict[str, Any]
    historical_summary: str
    seasonal_context: str
    budget_scenario: str
    response_curves: Dict[str, Any]

    # ─── Intermediate Analysis (Pure Python) ───
    performance_analysis: str
    anomalies: List[Dict[str, Any]]
    confidence_flags: List[Dict[str, Any]]

    # ─── Intermediate LLM Analysis ───
    anomaly_explanations: str
    risk_amplification: str

    # ─── Final Output ───
    summary: str
    key_drivers: List[str]
    risks: List[str]
    opportunities: List[str]
    anomaly_flags: List[str]
    final_output: Dict[str, Any]
