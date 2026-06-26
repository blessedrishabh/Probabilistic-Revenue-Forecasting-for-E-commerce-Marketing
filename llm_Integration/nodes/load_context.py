import json
import os
import sys

# Add the parent directory to sys.path to allow importing from Forecasting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm_Integration.state import CausalAnalysisState
# format_historical_summary import removed
# Import from forecasting module (assuming feature_engineering is accessible)
# If not, we can read raw data. For simplicity, we assume we load the precomputed CSV if possible,
# or we'll mock it if it's not directly importable.
# The plan mentions re-running the pipeline, but it's more robust to just read the raw csvs or assume the pipeline exports features.
# Let's write the node assuming it reads the forecast output JSON for now.

def load_context(state: CausalAnalysisState) -> dict:
    """
    Node 1: Loads the required data into the state.
    """
    # Load forecast output
    forecast_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Forecasting', 'forecast_output.json')
    try:
        with open(forecast_path, 'r') as f:
            forecast_data = json.load(f)
    except FileNotFoundError:
        forecast_data = {"error": "Forecast output not found"}

    # Mock historical summary for now, ideally we run `compute_features` and use `format_historical_summary`.
    # Let's populate it with a placeholder that the LLM can use.
    historical_summary = "### Last 12 Weeks Performance\n(Placeholder. Pipeline integration will provide real data.)"
    
    # Seasonal context
    seasonal_context = "Upcoming holidays in window: July 4th (Independence Day)."
    
    # Budget scenario
    scenario_name = forecast_data.get("scenario", "Baseline")
    total_budget = forecast_data.get("total_budget_input", 0)
    budget_scenario = f"Scenario: {scenario_name} | Total Budget: ${total_budget:,.2f}"
    
    # Marginal ROAS data (placeholder)
    response_curves = {}
    
    return {
        "forecast_data": forecast_data,
        "historical_summary": historical_summary,
        "seasonal_context": seasonal_context,
        "budget_scenario": budget_scenario,
        "response_curves": response_curves
    }
