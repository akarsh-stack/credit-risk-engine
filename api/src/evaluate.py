"""
src/evaluate.py
===============
Evaluation script for all three trained models.

Metrics computed:
  - ROC-AUC
  - PR-AUC (Average Precision)
  - KS Statistic (max separation between cumulative good/bad distributions)
  - F1, Precision, Recall at a chosen operating threshold (default 0.5)
  - Gini coefficient (= 2 * AUC - 1)

Outputs:
  - ROC curve comparison plot  -> models/plots/roc_curves.png
  - Calibration curve plot     -> models/plots/calibration_curve.png
  - Markdown comparison table  -> models/evaluation_report.md

Usage:
    python src/evaluate.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import load_and_split

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.35  # Credit risk: recall matters more -> lower threshold


# ── KS Statistic ────────────────────────────────────────────────────────────────
def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max vertical separation between
    the cumulative distributions of predicted probabilities for
    good loans (y=0) and bad loans (y=1).

    Standard credit risk metric, also equal to max(TPR - FPR) from ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ── Full metric suite ───────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "ROC-AUC":    round(auc, 4),
        "Gini":       round(2 * auc - 1, 4),
        "PR-AUC":     round(average_precision_score(y_true, y_prob), 4),
        "KS":         round(ks_statistic(y_true, y_prob), 4),
        "F1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Threshold":  threshold,
    }


# ── ROC Curve Plot ──────────────────────────────────────────────────────────────
def plot_roc_curves(models_probs: dict, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = {"Logistic Regression": "#60a5fa", "Random Forest": "#34d399", "XGBoost": "#f59e0b"}

    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=colors[name], linewidth=2.5)

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#475569")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")

    path = PLOTS_DIR / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved ROC plot -> {path}")


# ── Calibration Curve ───────────────────────────────────────────────────────────
def plot_calibration(y_test: np.ndarray, y_prob: np.ndarray, model_name: str = "XGBoost"):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=15, strategy="uniform"
    )
    ax.plot(mean_predicted_value, fraction_of_positives,
            "o-", color="#f59e0b", linewidth=2.5, markersize=6, label=f"{model_name} (calibrated)")
    ax.plot([0, 1], [0, 1], "w--", linewidth=1.5, alpha=0.6, label="Perfect calibration")

    ax.fill_between(mean_predicted_value, fraction_of_positives, mean_predicted_value,
                    alpha=0.1, color="#f59e0b")

    ax.set_xlabel("Mean Predicted Probability", color="white", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Actual Default Rate)", color="white", fontsize=12)
    ax.set_title(f"Calibration Curve — {model_name}", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#475569")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    path = PLOTS_DIR / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved calibration plot -> {path}")


# ── Markdown Report ─────────────────────────────────────────────────────────────
def write_markdown_report(results: dict[str, dict], threshold: float):
    lines = [
        "# Model Evaluation Report\n",
        f"**Operating threshold:** {threshold}  \n",
        f"**Dataset:** Test set (20% hold-out, stratified)\n",
        "",
        "## Metric Comparison\n",
        "| Model | ROC-AUC | Gini | PR-AUC | KS | F1 | Precision | Recall |",
        "|-------|---------|------|--------|----|----|-----------|--------|",
    ]
    for model, m in results.items():
        lines.append(
            f"| {model} | {m['ROC-AUC']} | {m['Gini']} | {m['PR-AUC']} "
            f"| {m['KS']} | {m['F1']} | {m['Precision']} | {m['Recall']} |"
        )

    lines += [
        "",
        "## Metric Definitions\n",
        "- **ROC-AUC**: Area under the Receiver Operating Characteristic curve. 0.5 = random, 1.0 = perfect.",
        "- **Gini**: 2×AUC − 1. Standard credit scoring metric; >0.40 is considered good for credit risk.",
        "- **PR-AUC**: Area under the Precision-Recall curve. More informative than ROC-AUC under class imbalance.",
        "- **KS**: Kolmogorov-Smirnov statistic. Maximum separation between cumulative good/bad distributions. "
          "KS > 0.40 is considered good in credit scoring.",
        "- **F1**: Harmonic mean of Precision and Recall at the chosen threshold.",
        "- **Precision**: Of predicted defaults, fraction that are true defaults.",
        "- **Recall**: Of actual defaults, fraction correctly identified.",
    ]

    report_path = MODELS_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved report -> {report_path}")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    _, _, X_test, _, _, y_test = load_and_split()
    y_test_arr = y_test.values

    # Load models
    models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "lr_model.joblib"),
        "Random Forest":       joblib.load(MODELS_DIR / "rf_model.joblib"),
        "XGBoost":             joblib.load(MODELS_DIR / "xgb_pipeline.joblib"),
    }

    results = {}
    probs_dict = {}

    print(f"\nEvaluating on test set ({len(X_test):,} samples, threshold={THRESHOLD})...")
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        probs_dict[name] = probs
        metrics = compute_metrics(y_test_arr, probs, threshold=THRESHOLD)
        results[name] = metrics
        print(f"\n  {name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    # Plots
    print("\nGenerating plots...")
    plot_roc_curves(probs_dict, y_test_arr)
    plot_calibration(y_test_arr, probs_dict["XGBoost"])

    # Markdown report
    report = write_markdown_report(results, THRESHOLD)
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    # Print the table section to console
    for line in report.split("\n")[3:9]:
        print(line)

    print("\n[OK] Evaluation complete")


if __name__ == "__main__":
    main()
