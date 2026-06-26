from datetime import datetime
from llm_Integration import config
from llm_Integration.state import CausalAnalysisState

def synthesize_output(state: CausalAnalysisState) -> dict:
    """
    Node 9: Synthesize all state parts into the final structured JSON format.
    """
    forecast_data = state.get("forecast_data", {})
    
    confidence_level = "HIGH"
    if len(state.get("confidence_flags", [])) > 0:
        confidence_level = "LOW"
    elif len(state.get("anomalies", [])) > 0:
        confidence_level = "MEDIUM"

    final_output = {
        "summary": state.get("summary", ""),
        "key_drivers": state.get("key_drivers", []),
        "risks": state.get("risks", []),
        "opportunities": state.get("opportunities", []),
        "anomalies": state.get("anomaly_flags", []),
        "confidence_level": confidence_level,
        "metadata": {
            "forecast_period_days": forecast_data.get("forecast_period_days", 30),
            "scenario": forecast_data.get("scenario", "Baseline"),
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "model_used": config.MODEL_NAME
        }
    }
    
    return {
        "final_output": final_output
    }
