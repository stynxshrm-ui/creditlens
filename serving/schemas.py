"""
serving/schemas.py

Pydantic request and response schemas for the CreditLens API.
These form the contract between the serving layer and consumers.
Changing these is a breaking change — version the API accordingly.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskTier(str, Enum):
    A = "A"  # Gini band 1 — lowest risk
    B = "B"
    C = "C"
    D = "D"
    E = "E"  # Gini band 5 — highest risk


class UpliftSegment(str, Enum):
    persuadable = "persuadable"
    sure_thing  = "sure_thing"
    lost_cause  = "lost_cause"
    safe        = "safe"
    unknown     = "unknown"   # borrower not in struggling pool


class ScoreRequest(BaseModel):
    """Features required to score a single loan application."""

    # Loan characteristics
    loan_amnt:    float = Field(..., gt=0, description="Loan amount in USD")
    int_rate:     float = Field(..., gt=0, lt=100, description="Interest rate %")
    term_months:  int   = Field(..., ge=36, le=60, description="Loan term in months")
    installment:  float = Field(..., gt=0, description="Monthly installment USD")
    purpose:      Optional[str] = None

    # Borrower financials
    annual_inc:   float = Field(..., gt=0, description="Annual income USD")
    dti:          float = Field(..., ge=0, description="Debt-to-income ratio %")
    revol_util:   Optional[float] = Field(None, ge=0, le=200)
    revol_bal:    Optional[float] = Field(None, ge=0)

    # Bureau history
    delinq_2yrs:      Optional[float] = Field(None, ge=0)
    open_acc:         Optional[float] = Field(None, ge=0)
    pub_rec:          Optional[float] = Field(None, ge=0)
    total_acc:        Optional[float] = Field(None, ge=0)
    emp_length_years: Optional[float] = Field(None, ge=0)
    grade_numeric:    Optional[int]   = Field(None, ge=1, le=7)
    home_ownership_enc: Optional[int] = Field(None, ge=0, le=3)

    # Payment behaviour (months 1-6)
    avg_payment_ratio_m6:   Optional[float] = None
    min_payment_ratio_m6:   Optional[float] = None
    payment_volatility_m6:  Optional[float] = None
    active_months_m6:       Optional[float] = None
    payment_trend_m6:       Optional[float] = None
    first_month_ratio:      Optional[float] = None
    last_month_ratio:       Optional[float] = None
    avg_monthly_change:     Optional[float] = None

    # Risk flags
    high_dti_flag:      Optional[int] = Field(None, ge=0, le=1)
    high_util_flag:     Optional[int] = Field(None, ge=0, le=1)
    dual_stress_flag:   Optional[int] = Field(None, ge=0, le=1)
    prior_delinq_flag:  Optional[int] = Field(None, ge=0, le=1)
    public_record_flag: Optional[int] = Field(None, ge=0, le=1)
    underpayment_flag:  Optional[int] = Field(None, ge=0, le=1)
    missed_payment_flag: Optional[int] = Field(None, ge=0, le=1)
    deteriorating_flag: Optional[int] = Field(None, ge=0, le=1)
    early_dropout_flag: Optional[int] = Field(None, ge=0, le=1)

    # Derived
    loan_to_income:     Optional[float] = None
    payment_to_income:  Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt":             15000,
                "int_rate":              14.85,
                "term_months":           36,
                "installment":           519.88,
                "annual_inc":            75000,
                "dti":                   18.5,
                "revol_util":            45.2,
                "avg_payment_ratio_m6":  0.98,
                "first_month_ratio":     1.02,
                "min_payment_ratio_m6":  0.91,
            }
        }


class SHAPContribution(BaseModel):
    feature: str
    value:   float
    shap:    float


class ScoreResponse(BaseModel):
    """Full scoring response for a single loan."""

    # Primary outputs
    pd_score:     float      = Field(..., description="Probability of default (0-1)")
    credit_score: int        = Field(..., description="Scorecard points (300-850)")
    risk_tier:    RiskTier   = Field(..., description="Risk band A-E")

    # Uplift outputs
    uplift_score: Optional[float]        = Field(None, description="Restructuring uplift score")
    segment:      Optional[UpliftSegment] = Field(None, description="Intervention segment")

    # Explainability
    shap_values:  Optional[list[SHAPContribution]] = Field(
        None, description="Top 5 SHAP feature contributions"
    )

    # Metadata
    model_version: str
    champion_model: str


class ModelInfo(BaseModel):
    champion_model:   str
    model_version:    str
    training_vintage: str
    gini:             float
    ks:               float
    calibration_error: float
    registered_at:    str


class HealthResponse(BaseModel):
    status:       str
    models_loaded: bool
    version:      str