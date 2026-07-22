# Model Evaluation Report

**Operating threshold:** 0.35  

**Dataset:** Test set (20% hold-out, stratified)


## Metric Comparison

| Model | ROC-AUC | Gini | PR-AUC | KS | F1 | Precision | Recall |
|-------|---------|------|--------|----|----|-----------|--------|
| Logistic Regression | 0.8357 | 0.6714 | 0.5629 | 0.5033 | 0.4961 | 0.3457 | 0.878 |
| Random Forest | 0.8346 | 0.6692 | 0.5608 | 0.5068 | 0.4911 | 0.3398 | 0.8851 |
| XGBoost | 0.8354 | 0.6708 | 0.5632 | 0.5059 | 0.5364 | 0.4121 | 0.7682 |

## Metric Definitions

- **ROC-AUC**: Area under the Receiver Operating Characteristic curve. 0.5 = random, 1.0 = perfect.
- **Gini**: 2×AUC − 1. Standard credit scoring metric; >0.40 is considered good for credit risk.
- **PR-AUC**: Area under the Precision-Recall curve. More informative than ROC-AUC under class imbalance.
- **KS**: Kolmogorov-Smirnov statistic. Maximum separation between cumulative good/bad distributions. KS > 0.40 is considered good in credit scoring.
- **F1**: Harmonic mean of Precision and Recall at the chosen threshold.
- **Precision**: Of predicted defaults, fraction that are true defaults.
- **Recall**: Of actual defaults, fraction correctly identified.