"""
train.py

Retrains the AMLGuard ensemble (XGBoost + Isolation Forest) on the latest
validated transaction data. Saves the trained model and its evaluation
metrics so evaluate.py can compare against the production baseline.

This assumes validate_data.py has already passed for this data.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "data/transactions.csv"
MODEL_OUTPUT_PATH = "artifacts/model.joblib"
METRICS_OUTPUT_PATH = "artifacts/metrics.json"

# Ensure the output directory exists before saving downstream artifacts
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

FEATURE_COLUMNS = [
    "transaction_amount",
    "transaction_hour",
    "is_new_counterparty",
    # TODO: add the rest of your real AMLGuard feature set here
]
LABEL_COLUMN = "label"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def train_models(X_train, y_train):
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    iso_forest.fit(X_train)

    return xgb_model, iso_forest


def evaluate(xgb_model, X_test, y_test) -> dict:
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = xgb_model.predict(X_test)

    return {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "n_test_samples": len(y_test),
    }


def main():
    print("Loading validated data...")
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)}...")
    xgb_model, iso_forest = train_models(X_train, y_train)

    metrics = evaluate(xgb_model, X_test, y_test)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump({"xgb_model": xgb_model, "iso_forest": iso_forest}, MODEL_OUTPUT_PATH)

    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

# Save a reference sample of the training data's feature distribution.
# check_drift.py compares new/incoming data against this to detect
# when live data has shifted enough to warrant retraining.
    BASELINE_SAMPLE_PATH = "artifacts/baseline_sample.csv"
    sample_size = min(500, len(X_train))
    X_train.sample(n=sample_size, random_state=42).to_csv(BASELINE_SAMPLE_PATH, index=False)
    print(f"Saved baseline distribution sample ({sample_size} rows) to {BASELINE_SAMPLE_PATH} for future drift detection.")

    print("Training complete.")


if __name__ == "__main__":
    main()