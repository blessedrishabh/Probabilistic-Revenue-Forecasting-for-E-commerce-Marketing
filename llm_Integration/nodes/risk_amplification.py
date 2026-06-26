from llm_Integration.state import CausalAnalysisState
from llm_Integration.utils.llm_client import get_llm, invoke_structured
from llm_Integration.prompts.output_schemas import RiskAmplificationOutput
import json

def risk_amplification(state: CausalAnalysisState) -> dict:
    flags = state.get("confidence_flags", [])
    if not flags:
        return {"risk_amplification": "High confidence in forecast intervals."}

    prompt = f"""The following segments have unusually wide forecast uncertainty bands:
{json.dumps(flags, indent=2)}

Performance context:
{state.get('performance_analysis', '')}

Identify:
1. Which specific factors (data volume, volatility, trend breaks) are driving the wide intervals.
2. Which weeks in the forecast window carry the most risk.
3. A recommended range interpretation for stakeholders.
"""
    llm = get_llm()
    result: RiskAmplificationOutput = invoke_structured(llm, RiskAmplificationOutput, prompt)
    
    details_str = "\n".join(f"- {d}" for d in result.risk_details)
    amplification = f"Risks:\n{details_str}\n\nGuidance: {result.stakeholder_guidance}"
    
    return {"risk_amplification": amplification}
