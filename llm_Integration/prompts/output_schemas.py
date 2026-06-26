from pydantic import BaseModel, Field
from typing import List

class AnomalyExplanation(BaseModel):
    anomaly_id: str = Field(description="Identifier or name of the anomaly/segment.")
    cause: str = Field(description="Proposed causal explanation for this anomaly.")
    confidence_impact: str = Field(description="Impact on forecast confidence: HIGH, MEDIUM, or LOW.")
    recommendation: str = Field(description="Recommendation for interpreting the forecast.")

class DeepAnomalyOutput(BaseModel):
    explanations: List[AnomalyExplanation]

class RiskAmplificationOutput(BaseModel):
    risk_details: List[str] = Field(description="Specific factors driving the wide uncertainty intervals.")
    stakeholder_guidance: str = Field(description="Recommended range interpretation for stakeholders.")

class CausalSummaryOutput(BaseModel):
    summary: str = Field(description="3-5 sentences explaining why revenue is expected to reach the P50 range.")
    key_drivers: List[str] = Field(description="Top 2 channel or campaign-type drivers.")
    risks: List[str] = Field(description="2 operational risks that could push outcome to P10.")
    anomalies: List[str] = Field(description="Any anomalies that reduce forecast confidence.")

class BudgetOpportunitiesOutput(BaseModel):
    opportunities: List[str] = Field(description="1-2 budget reallocation opportunities to improve P50 revenue.")
