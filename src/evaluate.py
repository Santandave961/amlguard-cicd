"""
evaluate.py

The gate that decides whether a newly retrained model is allowed to be
promoted. A training run finishing without errors does NOT mean the
resulting model is actually good — this script is what actually checks
that, by comparing against the current production baseline.

Exits non-zero (failing the pipeline) if the new model doesn't meet or
beat the baseline within the allowed tolerance.
"""

import json
import sys

METRICS_PATH = "artifacts/metrics.json"
BASELINE_PATH = "artifacts/baseline_metrics.json"

# Current AMLGuard production baseline (from the authoritative model record)
DEFAULT_BASELINE = {
    "roc_auc": 0.9914,
    "precision": 0.765,
    "recall": 0.9361,
    "f1": 0.8419,
}

# How much worse the new model is allowed to be before the pipeline fails.
# Small tolerance accounts for normal run-to-run variance.
TOLERANCE = {
    "roc_auc": 0.01,
    "precision": 0.03,
    "recall": 0.03,
    "f1": 0.03,
}


def load_baseline() -> dict:
    try:
        with open(BASELINE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No baseline file found at {BASELINE_PATH}, using DEFAULT_BASELINE.")
        return DEFAULT_BASELINE


def main():
    with open(METRICS_PATH) as f:
        new_metrics = json.load(f)

    baseline = load_baseline()

    print("Comparing new model against production baseline:")
    print(f"{'Metric':<12}{'Baseline':<12}{'New Model':<12}{'Result'}")

    failures = []
    for metric, baseline_value in baseline.items():
        new_value = new_metrics.get(metric)
        if new_value is None:
            continue

        tolerance = TOLERANCE.get(metric, 0.01)
        passed = new_value >= (baseline_value - tolerance)
        status = "PASS" if passed else "FAIL"
        print(f"{metric:<12}{baseline_value:<12.4f}{new_value:<12.4f}{status}")

        if not passed:
            failures.append(
                f"{metric}: new={new_value:.4f} is more than {tolerance} below "
                f"baseline={baseline_value:.4f}"
            )

    if failures:
        print("\nFAIL: new model does not meet production baseline.")
        for f_msg in failures:
            print(f"  - {f_msg}")
        print("\nProduction model will NOT be replaced. Pipeline halted.")
        sys.exit(1)

    print("\nPASS: new model meets or exceeds baseline. Safe to promote.")


if __name__ == "__main__":
    main()