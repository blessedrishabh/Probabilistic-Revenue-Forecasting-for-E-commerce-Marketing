from llm_Integration.state import CausalAnalysisState
from llm_Integration.utils.llm_client import get_llm, invoke_structured
from llm_Integration.prompts.output_schemas import CausalSummaryOutput
import json

def generate_causal_summary(state: CausalAnalysisState) -> dict:
    prompt = f"""SYSTEM:
You are an expert digital marketing analyst specializing in ecommerce performance forecasting. Analyze the structured data below and return a business-readable forecast explanation. Focus on causal reasoning, not generic statements.

USER:
## Historical Performance Summary
{state.get('historical_summary', '')}

## Detected Anomalies
{json.dumps(state.get('anomalies', []), indent=2)}
{state.get('anomaly_explanations', '')}

## Forecast Output
{json.dumps(state.get('forecast_data', {}), indent=2)}

## Seasonal Context
{state.get('seasonal_context', '')}

## Budget Scenario Applied
{state.get('budget_scenario', '')}

## Performance Signals
{state.get('performance_analysis', '')}

## Confidence Assessment  
{json.dumps(state.get('confidence_flags', []), indent=2)}
{state.get('risk_amplification', '')}

Task:
1. Explain in 3-5 sentences why revenue is expected to reach the P50 range.
2. Identify the top 2 channel or campaign-type drivers of this forecast.
3. List 2 operational risks that could push outcome toward P10.
4. Flag any anomalies in historical data that reduce forecast confidence.
"""
    llm = get_llm()
    result: CausalSummaryOutput = invoke_structured(llm, CausalSummaryOutput, prompt)
    
    return {
        "summary": result.summary,
        "key_drivers": result.key_drivers,
        "risks": result.risks,
        "anomaly_flags": result.anomalies
    }
