"""Pydantic schemas — type-safe request/response models for all endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    physicians_loaded: int
    features_loaded: int


class PhysicianBase(BaseModel):
    NPI: str
    specialty: str
    state: str
    adoption_score: float = Field(..., ge=0.0, le=1.0)
    predicted_days_to_rx: float
    urgency_tier: str
    urgency_tier_label: str
    patient_panel_size: int
    prior_injectable_glp1_prescriber: int
    kol_pagerank_normalized: float
    is_speaker: int
    total_open_payments_usd: float


class PhysicianDetail(PhysicianBase):
    years_in_practice: int
    gender: str
    rural: int
    num_speaker_events: int
    prior_sglt2_prescriber: int
    prior_dpp4_prescriber: int


class PredictRequest(BaseModel):
    state: Optional[str] = None
    specialty: Optional[str] = None
    min_score: float = 0.0
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class PredictResponse(BaseModel):
    physicians: List[PhysicianBase]
    total: int
    returned: int
    filters_applied: dict


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    abs_shap: float


class ExplainResponse(BaseModel):
    npi: str
    expected_value: float
    prediction_logit: float
    contributions: List[ShapContribution]


class TerritoryAggregate(BaseModel):
    state: str
    physician_count: int
    mean_adoption_score: float
    tier_1_count: int
    high_potential_count: int


class TerritoryResponse(BaseModel):
    territories: List[TerritoryAggregate]


class MetricsResponse(BaseModel):
    xgboost_auc: float
    xgboost_pr_auc: float
    xgboost_cv_mean: float
    xgboost_cv_std: float
    baseline_logistic_auc: float
    cox_concordance: float
    test_size: int
    test_positive_rate: float
    lift_by_decile: dict
    roc_curve: dict
    pr_curve: dict
    feature_importance: dict


class FilterOptions(BaseModel):
    states: List[str]
    specialties: List[str]
