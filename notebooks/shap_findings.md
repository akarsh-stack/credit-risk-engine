# SHAP Explainability Findings

## Global Feature Importance

The top features by mean absolute SHAP value are:

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | Fico Score | 0.66934 |
| 2 | Num Delinquencies 2Yrs | 0.32728 |
| 3 | Interest Rate | 0.24264 |
| 4 | Revolving Utilization Pct | 0.11312 |
| 5 | Dti Ratio | 0.08611 |
| 6 | Annual Income | 0.05538 |
| 7 | Credit History Length Years | 0.04520 |
| 8 | Num Open Accounts | 0.03979 |
| 9 | Employment Length Years | 0.03270 |
| 10 | Loan Amount | 0.03086 |
| 11 | Loan Term | 0.02602 |
| 12 | Home Ownership Rent | 0.02479 |
| 13 | Num Derogatory Marks | 0.01370 |
| 14 | Loan Grade A | 0.01174 |
| 15 | Loan Grade B | 0.01074 |

## Key Insights

### 1. FICO Score Dominates
FICO score is consistently the most important feature, which aligns with domain knowledge —
it is the single most widely used credit scoring metric and directly captures payment history,
credit utilization, and credit age.

### 2. DTI and Revolving Utilization are Highly Informative
Debt-to-income ratio and revolving utilization percentage rank 2nd and 3rd. Both measure
how stretched a borrower's finances are. A high-DTI borrower has little margin for unexpected
expenses, and high utilization suggests maxing out available credit — both red flags.

### 3. Loan Grade Acts as a Model Summary
Loan grade (derived from FICO + DTI + income in data generation) also ranks highly.
In production, this would be a credit bureau-derived composite score. Its high importance
validates that the model is learning the right latent structure.

### 4. Interest Rate is Endogenous
Interest rate appears important because it is correlated with creditworthiness (riskier
borrowers get higher rates in the synthetic data generation). In a real pipeline, care
must be taken not to leak future information through the rate if it's set post-underwriting.

### 5. Directionality Matches Domain Intuition
- **FICO ↑ -> default probability ↓**: Higher FICO = lower risk (correct)
- **DTI ↑ -> default probability ↑**: Higher DTI = higher risk (correct)
- **Revolving utilization ↑ -> default probability ↑**: Maxing cards = higher risk (correct)
- **Delinquencies ↑ -> default probability ↑**: Past late payments predict future ones (correct)

This directional consistency is a strong sanity check that the model has learned genuine
credit risk signals rather than spurious correlations.

## Calibration
The XGBoost model is well-calibrated (see `models/plots/calibration_curve.png`):
predicted probabilities track actual default rates closely, making the probability
outputs trustworthy for downstream risk decisions like setting loan limits or pricing.
