from llm_Integration.state import CausalAnalysisState

def analyze_performance(state: CausalAnalysisState) -> dict:
    """
    Node 2: Precomputes performance trends and flags based on structured data
    so the LLM can reason over facts.
    """
    forecast_data = state.get("forecast_data", {})
    
    analysis_lines = []
    
    # Example logic iterating over the forecast payload
    channels = forecast_data.get("channels", {})
    for channel, c_data in channels.items():
        budget = c_data.get("budget_allocated", 0)
        p50_rev = c_data.get("revenue", {}).get("P50", 0)
        p50_roas = c_data.get("roas", {}).get("P50", 0)
        
        analysis_lines.append(f"- **{channel.title()} Overall:** Budget ${budget:,.2f} | P50 Revenue ${p50_rev:,.2f} | P50 ROAS {p50_roas}x")
        
        for ct in c_data.get("campaign_types", []):
            ct_name = ct.get("campaign_type", "Unknown")
            ct_budget = ct.get("budget_allocated", 0)
            ct_p50_rev = ct.get("revenue", {}).get("P50", 0)
            
            if ct_budget > 0 and ct_p50_rev < 10:
                analysis_lines.append(f"  - ⚠️ {ct_name} shows near-zero projected revenue despite a budget of ${ct_budget:,.2f}.")
    
    performance_analysis = "\n".join(analysis_lines)
    
    return {
        "performance_analysis": performance_analysis
    }
