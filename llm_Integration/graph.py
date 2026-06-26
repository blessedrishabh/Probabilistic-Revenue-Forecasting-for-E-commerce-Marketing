from langgraph.graph import StateGraph, START, END
from llm_Integration.state import CausalAnalysisState
from llm_Integration.nodes.load_context import load_context
from llm_Integration.nodes.analyze_performance import analyze_performance
from llm_Integration.nodes.detect_anomalies import detect_anomalies
from llm_Integration.nodes.deep_anomaly_analysis import deep_anomaly_analysis
from llm_Integration.nodes.assess_confidence import assess_confidence
from llm_Integration.nodes.risk_amplification import risk_amplification
from llm_Integration.nodes.generate_causal_summary import generate_causal_summary
from llm_Integration.nodes.budget_opportunities import budget_opportunities
from llm_Integration.nodes.synthesize_output import synthesize_output

def should_analyze_anomalies(state: CausalAnalysisState) -> str:
    if len(state.get("anomalies", [])) > 0:
        return "deep_anomaly_analysis"
    return "assess_confidence"

def should_amplify_risk(state: CausalAnalysisState) -> str:
    if len(state.get("confidence_flags", [])) > 0:
        return "risk_amplification"
    return "generate_causal_summary"

def build_graph():
    workflow = StateGraph(CausalAnalysisState)
    
    # Add nodes
    workflow.add_node("load_context", load_context)
    workflow.add_node("analyze_performance", analyze_performance)
    workflow.add_node("detect_anomalies", detect_anomalies)
    workflow.add_node("deep_anomaly_analysis", deep_anomaly_analysis)
    workflow.add_node("assess_confidence", assess_confidence)
    workflow.add_node("risk_amplification", risk_amplification)
    workflow.add_node("generate_causal_summary", generate_causal_summary)
    workflow.add_node("budget_opportunities", budget_opportunities)
    workflow.add_node("synthesize_output", synthesize_output)
    
    # Add edges
    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "analyze_performance")
    workflow.add_edge("analyze_performance", "detect_anomalies")
    
    # Conditional edge after anomaly detection
    workflow.add_conditional_edges("detect_anomalies", should_analyze_anomalies, {
        "deep_anomaly_analysis": "deep_anomaly_analysis",
        "assess_confidence": "assess_confidence"
    })
    workflow.add_edge("deep_anomaly_analysis", "assess_confidence")
    
    # Conditional edge after confidence assessment
    workflow.add_conditional_edges("assess_confidence", should_amplify_risk, {
        "risk_amplification": "risk_amplification",
        "generate_causal_summary": "generate_causal_summary"
    })
    workflow.add_edge("risk_amplification", "generate_causal_summary")
    
    workflow.add_edge("generate_causal_summary", "budget_opportunities")
    workflow.add_edge("budget_opportunities", "synthesize_output")
    workflow.add_edge("synthesize_output", END)
    
    return workflow.compile()
