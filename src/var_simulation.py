"""
src/var_simulation.py
=====================
Monte Carlo Value-at-Risk (VaR) simulation for a portfolio of loans.

Methodology:
  - Sample 500 loans from the test set
  - Use the XGBoost model's predicted default probabilities as per-loan PD
  - Loss-Given-Default (LGD) = 60% of loan_amount (standard Basel assumption)
  - Exposure-at-Default (EAD) = loan_amount
  - Expected Loss per loan = PD × LGD × EAD
  - Monte Carlo: 10,000 scenarios, each drawing independent Bernoulli defaults
  - Portfolio loss per scenario = sum(Bernoulli(PD_i) × LGD × EAD_i)

Risk Metrics:
  - VaR 95%: 95th percentile of simulated loss distribution
  - VaR 99%: 99th percentile of simulated loss distribution
  - ES 95% (CVaR): Mean of losses exceeding VaR 95%

Simplification note:
  Default correlations between loans are IGNORED (independent defaults assumed).
  This underestimates tail risk. See README → Future Work for the copula extension.

Outputs:
  - models/plots/var_distribution.png  — loss histogram with VaR/ES lines
  - models/var_results.json            — numeric results for API / frontend

Usage:
    python src/var_simulation.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import load_and_split

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_PORTFOLIO = 500
N_SCENARIOS = 10_000
LGD = 0.60
RNG_SEED = 42


def run_var_simulation(
    n_portfolio: int = N_PORTFOLIO,
    n_scenarios: int = N_SCENARIOS,
    lgd: float = LGD,
    seed: int = RNG_SEED,
) -> dict:
    rng = np.random.default_rng(seed)

    # ── Load model and test data ────────────────────────────────────────────────
    pipe = joblib.load(MODELS_DIR / "xgb_pipeline.joblib")
    _, _, X_test_raw, _, _, y_test = load_and_split()

    # Sample portfolio
    sample_idx = rng.choice(len(X_test_raw), size=n_portfolio, replace=False)
    X_portfolio = X_test_raw.iloc[sample_idx].reset_index(drop=True)
    y_actual = y_test.values[sample_idx]

    # ── Get loan amounts and predicted default probabilities ────────────────────
    loan_amounts = X_portfolio["loan_amount"].values.astype(float)
    pd_probs = pipe.predict_proba(X_portfolio)[:, 1]

    print(f"  Portfolio: {n_portfolio} loans")
    print(f"  Total exposure: ${loan_amounts.sum():,.0f}")
    print(f"  Mean PD: {pd_probs.mean():.1%}")
    print(f"  Expected loss (EL): ${(pd_probs * lgd * loan_amounts).sum():,.0f}")

    # ── Monte Carlo simulation ──────────────────────────────────────────────────
    print(f"\n  Running {n_scenarios:,} Monte Carlo scenarios...")

    # Shape: (n_scenarios, n_portfolio)
    # Each row is one scenario's Bernoulli draws
    default_matrix = rng.binomial(n=1, p=pd_probs, size=(n_scenarios, n_portfolio))

    # Loss per loan per scenario: default × LGD × EAD
    loss_matrix = default_matrix * lgd * loan_amounts  # broadcasting

    # Portfolio loss per scenario
    portfolio_losses = loss_matrix.sum(axis=1)

    # ── Risk metrics ────────────────────────────────────────────────────────────
    var_95 = float(np.percentile(portfolio_losses, 95))
    var_99 = float(np.percentile(portfolio_losses, 99))
    es_95 = float(portfolio_losses[portfolio_losses >= var_95].mean())
    mean_loss = float(portfolio_losses.mean())
    max_loss = float(portfolio_losses.max())
    std_loss = float(portfolio_losses.std())

    print(f"\n  ── Risk Metrics ──")
    print(f"  Mean Loss (EL):  ${mean_loss:>12,.0f}")
    print(f"  Std Dev:         ${std_loss:>12,.0f}")
    print(f"  VaR 95%:         ${var_95:>12,.0f}")
    print(f"  VaR 99%:         ${var_99:>12,.0f}")
    print(f"  ES 95% (CVaR):   ${es_95:>12,.0f}")
    print(f"  Max Scenario:    ${max_loss:>12,.0f}")

    # ── Plot loss distribution ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    n_bins = 80
    ax.hist(portfolio_losses / 1e6, bins=n_bins, color="#60a5fa", alpha=0.7,
            edgecolor="none", label=f"Portfolio Loss Distribution\n({n_scenarios:,} scenarios)")

    # VaR / ES lines
    ax.axvline(var_95 / 1e6, color="#f59e0b", linewidth=2.5, linestyle="--",
               label=f"VaR 95%: ${var_95/1e6:.2f}M")
    ax.axvline(var_99 / 1e6, color="#ef4444", linewidth=2.5, linestyle="-.",
               label=f"VaR 99%: ${var_99/1e6:.2f}M")
    ax.axvline(es_95 / 1e6, color="#a78bfa", linewidth=2.5, linestyle=":",
               label=f"ES 95%:  ${es_95/1e6:.2f}M")
    ax.axvline(mean_loss / 1e6, color="#34d399", linewidth=1.5, linestyle="-",
               label=f"Mean Loss: ${mean_loss/1e6:.2f}M", alpha=0.8)

    # Shade tail region beyond VaR 95%
    hist_vals, bin_edges = np.histogram(portfolio_losses / 1e6, bins=n_bins)
    for i in range(len(hist_vals)):
        if bin_edges[i] >= var_95 / 1e6:
            ax.bar(bin_edges[i], hist_vals[i], width=bin_edges[i+1]-bin_edges[i],
                   color="#ef4444", alpha=0.3, align="edge")

    ax.set_xlabel("Portfolio Loss ($M)", color="white", fontsize=13)
    ax.set_ylabel("Frequency", color="white", fontsize=13)
    ax.set_title(
        f"Portfolio Loss Distribution — Monte Carlo VaR Simulation\n"
        f"N={n_portfolio} loans, LGD={lgd:.0%}, {n_scenarios:,} scenarios | "
        f"Independence assumption (see README)",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#475569")
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    path = PLOTS_DIR / "var_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved VaR plot → {path}")

    # ── Save numeric results ────────────────────────────────────────────────────
    results = {
        "n_portfolio": n_portfolio,
        "n_scenarios": n_scenarios,
        "lgd": lgd,
        "total_exposure": float(loan_amounts.sum()),
        "mean_pd": float(pd_probs.mean()),
        "expected_loss": float((pd_probs * lgd * loan_amounts).sum()),
        "simulated_mean_loss": mean_loss,
        "std_loss": std_loss,
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "max_loss": max_loss,
        # Histogram data for frontend charting (200 bins)
        "loss_distribution_sample": portfolio_losses[
            np.random.default_rng(seed).choice(len(portfolio_losses), size=2000, replace=False)
        ].tolist(),
        "loan_amounts": loan_amounts.tolist(),
        "pd_probs": pd_probs.tolist(),
    }

    results_path = MODELS_DIR / "var_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results → {results_path}")

    return results


def main():
    print("=" * 60)
    print("PORTFOLIO VAR SIMULATION")
    print("=" * 60)
    run_var_simulation()
    print("\n✓ VaR simulation complete")


if __name__ == "__main__":
    main()
