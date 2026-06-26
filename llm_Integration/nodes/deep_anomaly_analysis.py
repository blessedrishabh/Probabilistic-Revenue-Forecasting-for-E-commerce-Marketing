from llm_Integration.state import CausalAnalysisState
from llm_Integration.utils.llm_client import get_llm, invoke_structured
from llm_Integration.prompts.output_schemas import DeepAnomalyOutput
import json

def deep_anomaly_analysis(state: CausalAnalysisState) -> dict:
    anomalies = state.get("anomalies", [])
    if not anomalies:
        return {"anomaly_explanations": "No anomalies detected."}
        
    prompt = f"""You are an expert digital marketing analyst. The following anomalies were detected in historical performance data for an ecommerce advertiser:
{json.dumps(anomalies, indent=2)}

Historical context:
{state.get('historical_summary', '')}

For each anomaly:
1. Propose the most likely causal explanation.
2. Rate confidence impact: HIGH, MEDIUM, or LOW.
3. Suggest whether forecast intervals should be interpreted conservatively.
"""
    llm = get_llm()
    result: DeepAnomalyOutput = invoke_structured(llm, DeepAnomalyOutput, prompt)
    
    # Format the result back into a string or save as structured list
    explanations_str = ""
    for ex in result.explanations:
        explanations_str += f"- {ex.anomaly_id}: {ex.cause} (Impact: {ex.confidence_impact}). {ex.recommendation}\n"
        
    return {"anomaly_explanations": explanations_str}
