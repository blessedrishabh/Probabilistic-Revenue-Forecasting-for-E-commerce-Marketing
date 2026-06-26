from llm_Integration.state import CausalAnalysisState

def detect_anomalies(state: CausalAnalysisState) -> dict:
    """
    Node 3: Detect anomalies in historical data or the forecast outputs.
    """
    forecast_data = state.get("forecast_data", {})
    anomalies = []
    
    # Basic rule-based anomaly detection over the forecast
    for channel, c_data in forecast_data.get("channels", {}).items():
        for ct in c_data.get("campaign_types", []):
            ct_name = ct.get("campaign_type", "Unknown")
            p50_rev = ct.get("revenue", {}).get("P50", 0)
            budget = ct.get("budget_allocated", 0)
            
            # Anomaly: Budget is allocated but revenue is near zero
            if budget > 500 and p50_rev < 10:
                anomalies.append({
                    "segment": f"{channel} - {ct_name}",
                    "type": "Zero Revenue Projection",
                    "severity": "HIGH",
                    "description": f"Allocated budget of ${budget:,.2f} yields near-zero P50 revenue (${p50_rev:,.2f})."
                })
                
    return {
        "anomalies": anomalies
    }
