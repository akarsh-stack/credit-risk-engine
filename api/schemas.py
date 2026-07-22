"""
api/schemas.py
==============
Pydantic v2 schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request schemas ─────────────────────────────────────────────────────────────

class LoanFeatures(BaseModel):
    """Single loan application features."""
    fico_score: int = Field(..., ge=300, le=850, description="FICO credit score (300-850)")
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    dti_ratio: float = Field(..., ge=0, le=60, description="Debt-to-income ratio (%)")
    loan_amount: float = Field(..., gt=0, description="Loan amount in USD")
    loan_term: int = Field(..., description="Loan term in months (36 or 60)")
    interest_rate: float = Field(..., ge=1, le=35, description="Annual interest rate (%)")
    employment_length_years: Optional[float] = Field(None, ge=0, le=50,
        description="Years of employment (null if unknown)")
    home_ownership: str = Field(..., description="RENT | OWN | MORTGAGE")
    loan_purpose: str = Field(..., description="debt_consolidation | credit_card | home_improvement | ...")
    credit_history_length_years: int = Field(..., ge=1, le=50,
        description="Length of credit history in years")
    num_delinquencies_2yrs: int = Field(..., ge=0, description="Delinquencies in past 2 years")
    revolving_utilization_pct: Optional[float] = Field(None, ge=0, le=100,
        description="Revolving credit utilization % (null if unknown)")
    loan_grade: str = Field(..., description="Loan grade A-G")
    verification_status: str = Field(..., description="Verified | Source Verified | Not Verified")
    num_open_accounts: int = Field(..., ge=0, description="Number of open credit accounts")
    num_derogatory_marks: int = Field(..., ge=0, description="Number of derogatory marks")

    model_config = {
        "json_schema_extra": {
            "example": {
                "fico_score": 720,
                "annual_income": 75000,
                "dti_ratio": 18.5,
                "loan_amount": 15000,
                "loan_term": 36,
                "interest_rate": 11.5,
                "employment_length_years": 5,
                "home_ownership": "MORTGAGE",
                "loan_purpose": "debt_consolidation",
                "credit_history_length_years": 12,
                "num_delinquencies_2yrs": 0,
                "revolving_utilization_pct": 28.0,
                "loan_grade": "B",
                "verification_status": "Verified",
                "num_open_accounts": 8,
                "num_derogatory_marks": 0,
            }
        }
    }


class PortfolioRequest(BaseModel):
    """Portfolio VaR request — optional list of loans."""
    loans: Optional[list[LoanFeatures]] = Field(
        None,
        description="List of loan features. If null, uses pre-computed test-set sample."
    )
    n_scenarios: int = Field(10_000, ge=1000, le=50_000,
        description="Number of Monte Carlo scenarios")
    lgd: float = Field(0.60, ge=0.01, le=1.0,
        description="Loss-Given-Default fraction")


# ── Response schemas ────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class PredictionResponse(BaseModel):
    default_probability: float
    risk_grade: str  # A=very low, B=low, C=medium, D=high, E=very high
    risk_grade_label: str
    top_shap_features: list[SHAPFeature]
    model_version: str = "xgboost_v1"


class PortfolioVaRResponse(BaseModel):
    n_loans: int
    total_exposure: float
    mean_pd: float
    expected_loss: float
    var_95: float
    var_99: float
    es_95: float
    loss_distribution: list[float]  # sample of losses for histogram
    lgd: float
    n_scenarios: int


class StressTestRequest(BaseModel):
    unemployment_shock_pct: float = Field(2.5, ge=0, le=10, description="Unemployment rate increase (%)")
    interest_rate_bump_bps: float = Field(150.0, ge=0, le=500, description="Interest rate increase (bps)")
    collateral_haircut_pct: float = Field(15.0, ge=0, le=50, description="Collateral / Housing haircut (%)")


class StressTestResponse(BaseModel):
    baseline_expected_loss: float
    stressed_expected_loss: float
    baseline_var_95: float
    stressed_var_95: float
    baseline_var_99: float
    stressed_var_99: float
    loss_increase_pct: float
    loss_distribution_baseline: list[float]
    loss_distribution_stressed: list[float]


class EntityProfile(BaseModel):
    entity_id: str
    name: str
    sector: str
    total_exposure: float
    pd_score: float
    risk_grade: str
    fico_score: int
    dti_ratio: float
    num_delinquencies: int
    open_lines: int
    status: str
    recent_events: list[str]


class LogEntry(BaseModel):
    timestamp: str
    level: str  # INFO | WARN | ALERT | SYSTEM
    module: str
    message: str

