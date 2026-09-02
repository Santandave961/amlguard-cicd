"""
log_to_registry.py

Logs the trained model, its metrics, and its parameters to MLflow. This is
what lets you answer, with certainty, "which model version is in
production, what data was it trained on, and how did it perform" instead
of guessing based on scattered files.

Only run this AFTER evaluate.py has passed — we don't want to register
models that failed the quality gate.

Tracking URI resolution:
- If MLFLOW_TRACKING_URI is set (e.g. pointing at a local `mlflow ui`
  server, or a hosted tracking server), runs are logged there.
- If it's not set (e.g. in CI without a server available), MLflow falls
  back to local file-based tracking under ./mlruns — the run still gets
  logged and is valid, just not visible in a shared UI.
"""

import json
import os
import joblib
import mlflow
import mlflow.sklearn

MODEL_PATH = "artifacts/model.joblib"
METRICS_PATH = "artifacts/metrics.json"

EXPERIMENT_NAME = "amlguard-fraud-detection"


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking server: {tracking_uri}")
    else:
        print("MLFLOW_TRACKING_URI not set — falling back to local ./mlruns tracking.")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    models = joblib.load(MODEL_PATH)

    with mlflow.start_run() as run:
        print(f"Logging run {run.info.run_id} to MLflow...")

        # Log evaluation metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log model parameters for lineage
        xgb_model = models["xgb_model"]
        mlflow.log_params(xgb_model.get_params())

        # Log the model artifacts themselves
        mlflow.sklearn.log_model(xgb_model, "xgb_model")
        mlflow.sklearn.log_model(models["iso_forest"], "isolation_forest")

        # Tag this run so it's easy to filter/find later
        mlflow.set_tag("project", "AMLGuard")
        mlflow.set_tag("stage", "candidate")  # promote to "production" manually or via a separate step

        print(f"Run logged successfully: {run.info.run_id}")
        print(f"View at: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()