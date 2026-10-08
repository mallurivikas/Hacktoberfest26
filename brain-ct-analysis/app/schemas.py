from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


CT_LABEL = Literal[
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
]


class RiskSignal(BaseModel):
    model_name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    risk_level: Literal["low", "moderate", "high", "unknown"]
    score: Optional[float] = Field(default=None, ge=0, le=1)
    summary: Optional[str] = Field(default=None, max_length=500)


class HealthModelSummary(BaseModel):
    source: str = Field(min_length=1, max_length=80)
    overall_health_score: Optional[float] = Field(default=None, ge=0, le=100)
    summary: Optional[str] = Field(default=None, max_length=1200)
    risk_signals: List[RiskSignal] = Field(default_factory=list, max_length=20)


class RankedLabel(BaseModel):
    label: CT_LABEL
    score: float = Field(ge=0, le=1)


class BrainCTAnalysis(BaseModel):
    request_id: str
    model: str
    device: Literal["cpu", "cuda"]
    input_type: Literal["brain_ct_2d_image", "brain_ct_dicom_slice"]
    highest_scoring_label: CT_LABEL
    any_hemorrhage_score: float = Field(ge=0, le=1)
    class_scores: Dict[CT_LABEL, float]
    ranked_labels: List[RankedLabel]
    labels_at_or_above_model_threshold: List[CT_LABEL]
    health_model_summary: Optional[HealthModelSummary] = None
    integrated_summary: str
    clinical_review_required: bool = True
    disclaimer: str = (
        "Research and demonstration only. This single-slice screening model is not a diagnosis, "
        "radiology report, triage decision, or medical advice. A qualified clinician must review "
        "the complete CT study and clinical context."
    )


class ErrorResponse(BaseModel):
    detail: str
