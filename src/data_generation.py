"""
src/data_generation.py
======================
Generates a synthetic but statistically realistic credit dataset of 80,000 rows.

Architecture:
  - A hidden latent "creditworthiness" score drives both observable features
    and the default probability, creating realistic multicollinearity
    (fico/dti/utilization all correlate with each other AND the target).
  - Target base rate: ~17% defaults (realistic real-world imbalance).
  - 2-3% missingness injected into employment_length_years and
    revolving_utilization_pct to make preprocessing meaningful.

Usage:
    python src/data_generation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG_SEED = 42
N_ROWS = 80_000

# ── Output path ────────────────────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "credit_data.csv"


def generate_credit_data(n: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── 1. Latent creditworthiness score (z-scored, higher = more creditworthy)
    #        σ = 1.0 gives enough spread; noise added later per feature
    creditworthiness = rng.normal(loc=0.0, scale=1.0, size=n)

    # ── 2. Derive observable features from creditworthiness + idiosyncratic noise

    # FICO score: 300-850, strong positive correlation with creditworthiness
    fico_raw = 575 + 90 * creditworthiness + rng.normal(0, 35, n)
    fico_score = np.clip(np.round(fico_raw).astype(int), 300, 850)

    # Annual income ($): log-normal base; higher creditworthiness → higher income
    income_log_mean = 11.0 + 0.35 * creditworthiness  # exp(11) ≈ $60k
    annual_income = np.round(np.exp(rng.normal(income_log_mean, 0.45, n)), -2)
    annual_income = np.clip(annual_income, 12_000, 500_000)

    # Loan amount ($): roughly 1-5× monthly income, some driven by need (creditworthiness)
    loan_amount_raw = rng.lognormal(mean=9.5 - 0.2 * creditworthiness, sigma=0.6, size=n)
    loan_amount = np.round(np.clip(loan_amount_raw, 1_000, 40_000), -2)

    # DTI ratio (%): debt-to-income; NEGATIVELY correlated with creditworthiness
    dti_raw = 18 - 5 * creditworthiness + rng.normal(0, 5, n)
    dti_ratio = np.round(np.clip(dti_raw, 0, 50), 2)

    # Revolving utilization (%): NEGATIVELY correlated with creditworthiness
    util_raw = 45 - 15 * creditworthiness + rng.normal(0, 15, n)
    revolving_utilization_pct = np.round(np.clip(util_raw, 0, 100), 1)

    # Loan term: 36 or 60 months
    loan_term = rng.choice([36, 60], size=n, p=[0.65, 0.35])

    # Interest rate: driven by creditworthiness (riskier → higher rate)
    rate_raw = 13 - 3.5 * creditworthiness + 1.5 * (loan_term == 60).astype(float) + rng.normal(0, 2, n)
    interest_rate = np.round(np.clip(rate_raw, 5.0, 30.0), 2)

    # Employment length: positively (weakly) correlated with creditworthiness
    emp_raw = 5 + 1.5 * creditworthiness + rng.normal(0, 3, n)
    employment_length_years = np.round(np.clip(emp_raw, 0, 40)).astype(int)

    # Credit history length: age proxy, moderate positive correlation
    hist_raw = 8 + 3 * creditworthiness + rng.normal(0, 4, n)
    credit_history_length_years = np.round(np.clip(hist_raw, 1, 40)).astype(int)

    # Number of delinquencies (past 2 years): negative correlation with creditworthiness
    delq_prob = 1 / (1 + np.exp(2 * creditworthiness))  # sigmoid
    num_delinquencies_2yrs = rng.binomial(n=5, p=delq_prob)

    # Revolving accounts open
    open_raw = 8 + 2 * creditworthiness + rng.normal(0, 3, n)
    num_open_accounts = np.round(np.clip(open_raw, 1, 30)).astype(int)

    # Derogatory marks: rare events, negative creditworthiness correlation
    derog_prob = np.clip(0.08 - 0.05 * creditworthiness / 2, 0.01, 0.4)
    num_derogatory_marks = rng.binomial(n=3, p=derog_prob)

    # Home ownership (categorical)
    # More creditworthy → more likely to own; less → rent
    p_own = np.clip(0.20 + 0.10 * creditworthiness, 0.05, 0.45)
    p_mortgage = np.clip(0.45 + 0.05 * creditworthiness, 0.25, 0.65)
    p_rent = np.clip(1 - p_own - p_mortgage, 0.05, 0.60)
    # Normalize
    total = p_own + p_mortgage + p_rent
    p_own /= total; p_mortgage /= total; p_rent /= total

    home_ownership = np.array([
        rng.choice(["OWN", "MORTGAGE", "RENT"], p=[p_own[i], p_mortgage[i], p_rent[i]])
        for i in range(n)
    ])

    # Loan purpose (categorical, weakly correlated with creditworthiness)
    purposes = ["debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "medical", "car", "vacation", "other"]
    # Higher-creditworthiness borrowers more likely to have productive purposes
    purpose_weights = np.array([0.35, 0.25, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02])
    loan_purpose = rng.choice(purposes, size=n, p=purpose_weights / purpose_weights.sum())

    # Verification status
    verif_statuses = ["Verified", "Source Verified", "Not Verified"]
    # Less creditworthy → less likely to be verified
    p_verified = np.clip(0.40 + 0.15 * creditworthiness, 0.15, 0.70)
    p_source = np.clip(0.30 - 0.05 * creditworthiness, 0.10, 0.45)
    p_not = np.clip(1 - p_verified - p_source, 0.05, 0.50)
    total = p_verified + p_source + p_not
    p_verified /= total; p_source /= total; p_not /= total

    verification_status = np.array([
        rng.choice(verif_statuses, p=[p_verified[i], p_source[i], p_not[i]])
        for i in range(n)
    ])

    # ── 3. Derive loan_grade from a formula combining FICO, DTI, income
    #        Grade A (best) → G (worst)
    grade_score = (
        (fico_score - 300) / 550 * 0.50  # normalized fico, weight 50%
        - dti_ratio / 50 * 0.25           # lower DTI is better, weight 25%
        + (np.log(annual_income) - np.log(12_000)) / (np.log(500_000) - np.log(12_000)) * 0.25
    )
    grade_score = np.clip(grade_score, 0, 1)
    # Data-driven quantile bins so every grade is populated and the default-rate-by-
    # grade relationship stays monotonic. Target mix ~ a real Lending Club book
    # (A ~16% of loans down to G ~1%), rather than fixed cutoffs that pile everyone
    # into the worst grades when the score distribution shifts.
    grade_quantiles = np.quantile(grade_score, [0.01, 0.04, 0.12, 0.28, 0.55, 0.84])
    grade_idx = np.digitize(grade_score, grade_quantiles)  # 0=worst score .. 6=best
    grade_labels = np.array(["G", "F", "E", "D", "C", "B", "A"])
    loan_grade = grade_labels[grade_idx]

    # ── 4. Default probability via logistic function of creditworthiness
    #        Add noise to prevent "too perfect" model (real AUC 0.70-0.80)
    noise = rng.normal(0, 0.7, n)  # idiosyncratic default risk
    default_logit = -2.2 - 1.8 * creditworthiness + 0.8 * noise
    default_prob = 1 / (1 + np.exp(-default_logit))

    # Draw actual defaults; targeting ~17% base rate
    default = rng.binomial(n=1, p=default_prob).astype(int)

    # ── 5. Assemble DataFrame
    df = pd.DataFrame({
        "fico_score": fico_score,
        "annual_income": annual_income,
        "dti_ratio": dti_ratio,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "employment_length_years": employment_length_years,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "credit_history_length_years": credit_history_length_years,
        "num_delinquencies_2yrs": num_delinquencies_2yrs,
        "revolving_utilization_pct": revolving_utilization_pct,
        "loan_grade": loan_grade,
        "verification_status": verification_status,
        "num_open_accounts": num_open_accounts,
        "num_derogatory_marks": num_derogatory_marks,
        "default": default,
    })

    # ── 6. Inject realistic missingness (2-3%) into two features
    miss_emp = rng.choice(n, size=int(0.025 * n), replace=False)
    miss_util = rng.choice(n, size=int(0.020 * n), replace=False)
    df.loc[miss_emp, "employment_length_years"] = np.nan
    df.loc[miss_util, "revolving_utilization_pct"] = np.nan

    return df


def main():
    print("Generating synthetic credit dataset...")
    df = generate_credit_data()

    # ── Sanity checks
    default_rate = df["default"].mean()
    print(f"  Rows:         {len(df):,}")
    print(f"  Features:     {df.shape[1] - 1}")
    print(f"  Default rate: {default_rate:.1%}")
    print(f"  Missing emp:  {df['employment_length_years'].isna().mean():.1%}")
    print(f"  Missing util: {df['revolving_utilization_pct'].isna().mean():.1%}")

    assert 0.13 <= default_rate <= 0.22, f"Default rate {default_rate:.1%} outside 13-22% — check noise params"

    # Correlation sanity check (fico should be negatively correlated with default)
    corr_fico = df["fico_score"].corr(df["default"])
    corr_dti = df["dti_ratio"].corr(df["default"])
    print(f"  Corr(fico, default): {corr_fico:.3f}  (expected < -0.3)")
    print(f"  Corr(dti,  default): {corr_dti:.3f}  (expected >  0.1)")
    assert corr_fico < -0.3, "FICO-default correlation too weak"
    assert corr_dti > 0.1, "DTI-default correlation too weak"

    # Grade monotonicity check (A should have lowest default rate, G highest)
    grade_rates = df.groupby("loan_grade")["default"].mean().sort_index()
    print("\n  Default rate by grade:")
    for g, r in grade_rates.items():
        print(f"    Grade {g}: {r:.1%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
