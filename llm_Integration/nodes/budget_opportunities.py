from llm_Integration.state import CausalAnalysisState
from llm_Integration.utils.llm_client import get_llm, invoke_structured
from llm_Integration.prompts.output_schemas import BudgetOpportunitiesOutput
import json

def budget_opportunities(state: CausalAnalysisState) -> dict:
    prompt = f"""Given the following response curves (marginal ROAS at current spend):
{json.dumps(state.get('response_curves', {}), indent=2)}

And the current forecast:
{json.dumps(state.get('forecast_data', {}), indent=2)}

Identify 1-2 budget reallocation opportunities that could improve P50 revenue.
For each, estimate the revenue impact of a 10% budget shift.
Be specific about source and destination channels/campaign types.
"""
    llm = get_llm()
    result: BudgetOpportunitiesOutput = invoke_structured(llm, BudgetOpportunitiesOutput, prompt)
    
    return {"opportunities": result.opportunities}
