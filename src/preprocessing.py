"""
src/preprocessing.py
====================
Preprocessing pipeline: imputation → one-hot encoding → scaling.
Produces stratified 60/20/20 train/val/test splits and saves the
fitted pipeline to models/preprocessor.joblib.

Usage:
    python src/preprocessing.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "credit_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# ── Feature definitions ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "fico_score",
    "annual_income",
    "dti_ratio",
    "loan_amount",
    "loan_term",
    "interest_rate",
    "employment_length_years",   # has ~2.5% missingness
    "credit_history_length_years",
    "num_delinquencies_2yrs",
    "revolving_utilization_pct", # has ~2.0% missingness
    "num_open_accounts",
    "num_derogatory_marks",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "loan_grade",
    "verification_status",
]

TARGET = "default"


def build_preprocessor() -> ColumnTransformer:
    """Construct the sklearn ColumnTransformer preprocessing pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fit."""
    return list(preprocessor.get_feature_names_out())


def load_and_split(data_path: Path = DATA_PATH):
    """Load raw data and produce stratified 60/20/20 splits."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # 60% train / 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    # 50% of temp → val (20% overall), 50% → test (20% overall)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Default rate — train: {y_train.mean():.1%}, val: {y_val.mean():.1%}, test: {y_test.mean():.1%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing():
    """Fit preprocessor on train set, transform all splits, save artifacts."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    preprocessor = build_preprocessor()

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"\nFeatures after preprocessing: {len(feature_names)}")

    # Convert to DataFrames and save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, X_proc, y in [
        ("train", X_train_processed, y_train),
        ("val", X_val_processed, y_val),
        ("test", X_test_processed, y_test),
    ]:
        df_proc = pd.DataFrame(X_proc, columns=feature_names)
        df_proc[TARGET] = y.values
        out_path = PROCESSED_DIR / f"{split_name}.csv"
        df_proc.to_csv(out_path, index=False)
        print(f"  Saved {split_name} split → {out_path}")

    # Save fitted preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n✓ Preprocessor saved → {preprocessor_path}")

    # Also save feature names for reference
    feature_names_path = MODELS_DIR / "feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)

    return preprocessor, feature_names


if __name__ == "__main__":
    run_preprocessing()
