from llm_Integration import config
from llm_Integration.state import CausalAnalysisState

def assess_confidence(state: CausalAnalysisState) -> dict:
    """
    Node 5: Assesses forecast uncertainty using the P90/P10 interval spread.
    """
    forecast_data = state.get("forecast_data", {})
    confidence_flags = []
    
    threshold = config.CONFIDENCE_INTERVAL_RATIO_THRESHOLD
    
    for channel, c_data in forecast_data.get("channels", {}).items():
        for ct in c_data.get("campaign_types", []):
            ct_name = ct.get("campaign_type", "Unknown")
            p10 = ct.get("revenue", {}).get("P10", 0)
            p50 = ct.get("revenue", {}).get("P50", 0)
            p90 = ct.get("revenue", {}).get("P90", 0)
            
            if p50 > 0:
                ratio = (p90 - p10) / p50
                if ratio > threshold:
                    confidence_flags.append({
                        "segment": f"{channel} - {ct_name}",
                        "ratio": round(ratio, 2),
                        "description": f"Interval width is {(ratio*100):.1f}% of P50. Indicates high uncertainty."
                    })
                    
    return {
        "confidence_flags": confidence_flags
    }
