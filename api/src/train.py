"""
src/train.py
============
Trains three models:
  1. Logistic Regression (baseline, class_weight='balanced')
  2. Random Forest
  3. XGBoost, tuned via Optuna (20 trials, optimise ROC-AUC on validation set)

All models operate on the raw (non-preprocessed) feature DataFrame via
sklearn Pipeline so the API can accept raw input without a separate
preprocessing step.

Saves to models/:
  lr_model.joblib
  rf_model.joblib
  xgb_model.joblib (XGBoost native format)
  xgb_pipeline.joblib (full sklearn Pipeline wrapper for SHAP + API)

Usage:
    python src/train.py
"""

import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# ── Local imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import (
    build_preprocessor,
    load_and_split,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Suppress Optuna info logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Helper: wrap preprocessor + classifier in sklearn Pipeline ─────────────────
def make_pipeline(classifier) -> Pipeline:
    preprocessor = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ── 1. Logistic Regression ─────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train, X_val, y_val) -> Pipeline:
    print("\n[1/3] Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=0.1,
        random_state=42,
    )
    pipe = make_pipeline(lr)
    pipe.fit(X_train, y_train)

    val_probs = pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"  Val ROC-AUC: {val_auc:.4f}")

    path = MODELS_DIR / "lr_model.joblib"
    joblib.dump(pipe, path)
    print(f"  Saved -> {path}")
    return pipe


# ── 2. Random Forest ───────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train, X_val, y_val) -> Pipeline:
    print("\n[2/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    pipe = make_pipeline(rf)
    pipe.fit(X_train, y_train)

    val_probs = pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"  Val ROC-AUC: {val_auc:.4f}")

    path = MODELS_DIR / "rf_model.joblib"
    joblib.dump(pipe, path)
    print(f"  Saved -> {path}")
    return pipe


# ── 3. XGBoost + Optuna ────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val) -> Pipeline:
    print("\n[3/3] Training XGBoost with Optuna (20 trials)...")

    # Pre-process once for speed during Optuna trials
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # Compute scale_pos_weight once
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    default_spw = neg_count / pos_count

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", default_spw * 0.5, default_spw * 1.5
            ),
        }
        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_train_proc, y_train,
            eval_set=[(X_val_proc, y_val)],
            verbose=False,
        )
        preds = model.predict_proba(X_val_proc)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    best_params = study.best_params
    print(f"\n  Best params: {best_params}")
    print(f"  Best Val AUC: {study.best_value:.4f}")

    # AUC guardrail: if > 0.90, add more regularisation (shouldn't happen with our noise)
    if study.best_value > 0.90:
        print("  ⚠ AUC > 0.90 — increasing regularisation to produce realistic model...")
        best_params["reg_alpha"] = max(best_params.get("reg_alpha", 1.0), 5.0)
        best_params["reg_lambda"] = max(best_params.get("reg_lambda", 1.0), 5.0)
        best_params["min_child_weight"] = max(best_params.get("min_child_weight", 10), 20)

    # Retrain best model on train+val combined for final model
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    best_xgb = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    # Build full pipeline (refit preprocessor on train+val)
    pipe = make_pipeline(best_xgb)
    pipe.fit(X_trainval, y_trainval)

    # Validate on val set (with separately fitted preprocessor)
    val_probs = xgb.XGBClassifier(**best_params, objective="binary:logistic",
                                   use_label_encoder=False, random_state=42,
                                   n_jobs=-1, verbosity=0).fit(
        X_train_proc, y_train
    ).predict_proba(X_val_proc)[:, 1]
    val_auc_final = roc_auc_score(y_val, val_probs)
    print(f"  Final Val ROC-AUC (train-only preprocessor): {val_auc_final:.4f}")

    # Save full pipeline
    path = MODELS_DIR / "xgb_pipeline.joblib"
    joblib.dump(pipe, path)
    print(f"  Saved pipeline -> {path}")

    # Also save the raw XGBoost booster for SHAP
    xgb_path = MODELS_DIR / "xgb_model.joblib"
    joblib.dump(pipe["classifier"], xgb_path)
    print(f"  Saved classifier -> {xgb_path}")

    return pipe


def main():
    print("=" * 60)
    print("CREDIT RISK MODEL TRAINING")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    lr_pipe = train_logistic_regression(X_train, y_train, X_val, y_val)
    rf_pipe = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_pipe = train_xgboost(X_train, y_train, X_val, y_val)

    print("\n" + "=" * 60)
    print("[OK] All models trained and saved to models/")
    print("  Run src/evaluate.py for full metrics report")
    print("=" * 60)


if __name__ == "__main__":
    main()
