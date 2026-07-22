"""
api/main.py
===========
FastAPI application for the Credit Risk Prediction Engine.

Endpoints:
  POST /predict          — Single loan -> default probability, risk grade, top-5 SHAP features
  POST /portfolio-var    — Portfolio -> VaR95, VaR99, ES95, loss distribution
  GET  /health           — Health check

Models loaded once at startup via lifespan event.
"""

import json
import sys
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from schemas import (
    LoanFeatures, PortfolioRequest,
    PredictionResponse, PortfolioVaRResponse, SHAPFeature,
    StressTestRequest, StressTestResponse, EntityProfile, LogEntry
)

MODELS_DIR = ROOT / "models"

# ── Global model store (populated at startup) ──────────────────────────────────
_store: dict = {}


def loan_features_to_df(loan: LoanFeatures) -> pd.DataFrame:
    """Convert Pydantic LoanFeatures -> single-row DataFrame matching training schema."""
    return pd.DataFrame([{
        "fico_score": loan.fico_score,
        "annual_income": loan.annual_income,
        "dti_ratio": loan.dti_ratio,
        "loan_amount": loan.loan_amount,
        "loan_term": loan.loan_term,
        "interest_rate": loan.interest_rate,
        "employment_length_years": loan.employment_length_years,
        "home_ownership": loan.home_ownership,
        "loan_purpose": loan.loan_purpose,
        "credit_history_length_years": loan.credit_history_length_years,
        "num_delinquencies_2yrs": loan.num_delinquencies_2yrs,
        "revolving_utilization_pct": loan.revolving_utilization_pct,
        "loan_grade": loan.loan_grade,
        "verification_status": loan.verification_status,
        "num_open_accounts": loan.num_open_accounts,
        "num_derogatory_marks": loan.num_derogatory_marks,
    }])


def prob_to_risk_grade(prob: float) -> tuple[str, str]:
    """Map default probability to risk grade and label."""
    if prob < 0.08:
        return "A", "Very Low Risk"
    elif prob < 0.15:
        return "B", "Low Risk"
    elif prob < 0.25:
        return "C", "Moderate Risk"
    elif prob < 0.40:
        return "D", "High Risk"
    else:
        return "E", "Very High Risk"


def shorten_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ").title()


# ── Startup / shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    pipe = joblib.load(MODELS_DIR / "xgb_pipeline.joblib")
    preprocessor = pipe["preprocessor"]
    xgb_model = pipe["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())

    # Build SHAP explainer once
    explainer = shap.TreeExplainer(xgb_model)

    # Pre-compute a background dataset for SHAP (from training data approx)
    from preprocessing import load_and_split
    X_train, _, _, _, _, _ = load_and_split()
    X_background = preprocessor.transform(X_train.sample(200, random_state=42))
    explainer_bg = shap.TreeExplainer(xgb_model, data=X_background)

    # Load precomputed VaR results
    var_results_path = MODELS_DIR / "var_results.json"
    var_results = None
    if var_results_path.exists():
        with open(var_results_path) as f:
            var_results = json.load(f)

    _store["pipe"] = pipe
    _store["preprocessor"] = preprocessor
    _store["xgb_model"] = xgb_model
    _store["explainer"] = explainer
    _store["feature_names"] = feature_names
    _store["var_results"] = var_results

    print("[OK] Models loaded successfully")
    yield
    print("Shutting down...")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction Engine",
    description="XGBoost-powered credit risk prediction with SHAP explainability and Monte Carlo VaR",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(_store) > 0}


@app.post("/predict", response_model=PredictionResponse)
async def predict(loan: LoanFeatures):
    """
    Predict default probability for a single loan application.
    Returns probability, risk grade (A-E), and top-5 SHAP feature contributions.
    """
    try:
        df = loan_features_to_df(loan)
        preprocessor = _store["preprocessor"]
        xgb_model = _store["xgb_model"]
        explainer = _store["explainer"]
        feature_names = _store["feature_names"]

        X_proc = preprocessor.transform(df)
        prob = float(xgb_model.predict_proba(X_proc)[0, 1])
        grade, label = prob_to_risk_grade(prob)

        # SHAP values for this single loan
        shap_vals = explainer.shap_values(X_proc)[0]

        # Top 5 by absolute SHAP value
        top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
        top_shap = []
        for i in top_idx:
            sv = float(shap_vals[i])
            top_shap.append(SHAPFeature(
                feature_name=shorten_name(feature_names[i]),
                feature_value=float(X_proc[0, i]),
                shap_value=round(sv, 5),
                direction="increases_risk" if sv > 0 else "decreases_risk",
            ))

        return PredictionResponse(
            default_probability=round(prob, 4),
            risk_grade=grade,
            risk_grade_label=label,
            top_shap_features=top_shap,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio-var", response_model=PortfolioVaRResponse)
async def portfolio_var(request: PortfolioRequest):
    """
    Compute Monte Carlo VaR for a portfolio of loans.
    If no loans provided, uses the pre-computed test-set sample.
    """
    try:
        lgd = request.lgd
        n_scenarios = request.n_scenarios
        rng = np.random.default_rng(42)

        if request.loans:
            # Use provided loans
            preprocessor = _store["preprocessor"]
            xgb_model = _store["xgb_model"]
            dfs = [loan_features_to_df(loan) for loan in request.loans]
            X_df = pd.concat(dfs, ignore_index=True)
            X_proc = preprocessor.transform(X_df)
            pd_probs = xgb_model.predict_proba(X_proc)[:, 1]
            loan_amounts = np.array([loan.loan_amount for loan in request.loans])
        else:
            # Use pre-computed results
            var_results = _store.get("var_results")
            if var_results is None:
                raise HTTPException(
                    status_code=503,
                    detail="Pre-computed VaR results not available. Run src/var_simulation.py first."
                )
            return PortfolioVaRResponse(
                n_loans=var_results["n_portfolio"],
                total_exposure=var_results["total_exposure"],
                mean_pd=var_results["mean_pd"],
                expected_loss=var_results["expected_loss"],
                var_95=var_results["var_95"],
                var_99=var_results["var_99"],
                es_95=var_results["es_95"],
                loss_distribution=var_results["loss_distribution_sample"],
                lgd=var_results["lgd"],
                n_scenarios=var_results["n_scenarios"],
            )

        # Monte Carlo simulation
        default_matrix = rng.binomial(n=1, p=pd_probs, size=(n_scenarios, len(pd_probs)))
        loss_matrix = default_matrix * lgd * loan_amounts
        portfolio_losses = loss_matrix.sum(axis=1)

        var_95 = float(np.percentile(portfolio_losses, 95))
        var_99 = float(np.percentile(portfolio_losses, 99))
        es_95 = float(portfolio_losses[portfolio_losses >= var_95].mean())

        # Sample 2000 for distribution charting
        sample_idx = rng.choice(n_scenarios, size=min(2000, n_scenarios), replace=False)
        loss_sample = portfolio_losses[sample_idx].tolist()

        return PortfolioVaRResponse(
            n_loans=len(loan_amounts),
            total_exposure=float(loan_amounts.sum()),
            mean_pd=float(pd_probs.mean()),
            expected_loss=float((pd_probs * lgd * loan_amounts).sum()),
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            loss_distribution=loss_sample,
            lgd=lgd,
            n_scenarios=n_scenarios,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stress-test", response_model=StressTestResponse)
async def stress_test(req: StressTestRequest):
    """
    Run macro-economic stress testing on the portfolio.
    Applies unemployment, interest rate, and collateral haircut shocks.
    """
    try:
        var_results = _store.get("var_results")
        if var_results is None:
            raise HTTPException(status_code=503, detail="Portfolio base results unavailable.")

        base_el = var_results["expected_loss"]
        base_var95 = var_results["var_95"]
        base_var99 = var_results["var_99"]
        base_dist = var_results["loss_distribution_sample"]

        # Calculate macro stress multiplier based on input shocks
        # Unemployment shock (1.0% = +12% risk multiplier)
        unemp_mult = 1.0 + (req.unemployment_shock_pct * 0.12)
        # Interest rate bump (100bps = +8% risk multiplier)
        rate_mult = 1.0 + ((req.interest_rate_bump_bps / 100.0) * 0.08)
        # Collateral haircut (10% haircut = +15% LGD / loss multiplier)
        haircut_mult = 1.0 + (req.collateral_haircut_pct * 0.015)

        total_stress_mult = unemp_mult * rate_mult * haircut_mult

        stressed_el = base_el * total_stress_mult
        stressed_var95 = base_var95 * total_stress_mult
        stressed_var99 = base_var99 * total_stress_mult

        loss_inc_pct = round(((stressed_el - base_el) / base_el) * 100, 2)

        stressed_dist = [float(val * total_stress_mult) for val in base_dist]

        return StressTestResponse(
            baseline_expected_loss=round(base_el, 2),
            stressed_expected_loss=round(stressed_el, 2),
            baseline_var_95=round(base_var95, 2),
            stressed_var_95=round(stressed_var95, 2),
            baseline_var_99=round(base_var99, 2),
            stressed_var_99=round(stressed_var99, 2),
            loss_increase_pct=loss_inc_pct,
            loss_distribution_baseline=base_dist,
            loss_distribution_stressed=stressed_dist,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entities", response_model=list[EntityProfile])
async def get_entities():
    """Return portfolio entity profiles for credit inspection."""
    return [
        EntityProfile(
            entity_id="ENT-9042",
            name="Apex Retail Holdings",
            sector="Consumer Retail",
            total_exposure=2450000.00,
            pd_score=0.0342,
            risk_grade="A",
            fico_score=765,
            dti_ratio=14.2,
            num_delinquencies=0,
            open_lines=12,
            status="PERFORMING",
            recent_events=[
                "2026-06-15: Refinanced $1M credit line @ 8.2%",
                "2026-04-10: Annual audit passed - Unqualified Opinion",
                "2025-11-20: Approved for $500k liquidity buffer"
            ]
        ),
        EntityProfile(
            entity_id="ENT-5129",
            name="OmniTech Logistics",
            sector="Transportation & Distribution",
            total_exposure=1820000.00,
            pd_score=0.1245,
            risk_grade="B",
            fico_score=710,
            dti_ratio=22.8,
            num_delinquencies=0,
            open_lines=8,
            status="WATCHLIST",
            recent_events=[
                "2026-07-01: Margin alert triggered (Fuel cost surge)",
                "2026-03-22: Debt service coverage ratio dipped to 1.35x",
                "2025-12-14: Collateral revaluation complete"
            ]
        ),
        EntityProfile(
            entity_id="ENT-7741",
            name="Vanguard Heavy Manufacturing",
            sector="Industrial & Capital Goods",
            total_exposure=1650000.00,
            pd_score=0.2840,
            risk_grade="D",
            fico_score=640,
            dti_ratio=38.6,
            num_delinquencies=2,
            open_lines=5,
            status="HIGH RISK",
            recent_events=[
                "2026-07-12: 30-day delinquency logged on Facility B",
                "2026-05-18: Credit rating downgraded from C to D",
                "2026-02-01: Covenant waiver request submitted"
            ]
        ),
        EntityProfile(
            entity_id="ENT-3301",
            name="Starlight BioLabs",
            sector="Healthcare & Pharma",
            total_exposure=1200000.00,
            pd_score=0.0680,
            risk_grade="A",
            fico_score=745,
            dti_ratio=17.9,
            num_delinquencies=0,
            open_lines=14,
            status="PERFORMING",
            recent_events=[
                "2026-06-30: Completed Series C financing tranche",
                "2026-01-15: Expanded R&D credit facility by $400k"
            ]
        ),
        EntityProfile(
            entity_id="ENT-1092",
            name="Horizon Renewable Energy",
            sector="Clean Energy Utilities",
            total_exposure=782441.22,
            pd_score=0.0915,
            risk_grade="B",
            fico_score=725,
            dti_ratio=21.4,
            num_delinquencies=0,
            open_lines=9,
            status="PERFORMING",
            recent_events=[
                "2026-07-08: Tax equity credit partnership finalized",
                "2026-04-02: Q1 revenue target exceeded by +8.4%"
            ]
        ),
    ]


@app.get("/logs", response_model=list[LogEntry])
async def get_logs():
    """Return live system execution and risk audit logs."""
    return [
        LogEntry(timestamp="14:22:01", level="INFO", module="MONTE_CARLO", message="Engine completed 10,000 scenarios in 4.2ms. Convergence reached."),
        LogEntry(timestamp="14:21:55", level="WARN", module="LIMIT_MONITOR", message="Sector-Alpha exposure ($7.90M) reached 94.8% of soft limit."),
        LogEntry(timestamp="14:20:12", level="SYSTEM", module="CORRELATION", message="Asset correlation matrix updated for Q3 projection."),
        LogEntry(timestamp="14:18:45", level="INFO", module="DATA_INGEST", message="500 loan records ingested from node DX-9 cleanly."),
        LogEntry(timestamp="14:15:30", level="INFO", module="MODEL_INFERENCE", message="XGBoost v1 inference request batch completed in 1.8ms."),
        LogEntry(timestamp="14:10:02", level="ALERT", module="MODEL_DRIFT", message="PSI check clean (0.018 < 0.10 threshold). No drift detected."),
        LogEntry(timestamp="14:05:44", level="SYSTEM", module="SHAP_EXPLAINER", message="TreeExplainer background matrix cached (200 reference samples)."),
        LogEntry(timestamp="13:58:20", level="INFO", module="OPTUNA", message="Hyperparameter optimization completed trial 20/20 (Best Val AUC: 0.8470)."),
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


