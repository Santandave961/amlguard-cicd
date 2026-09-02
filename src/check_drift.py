"""
check_drift.py

Compares the feature distributions of current/incoming transaction data
against the training data's reference distribution (saved by train.py as
artifacts/baseline_sample.csv). This is what closes the loop on the
"models degrade silently" problem — instead of waiting for a scheduled
retrain, this can run on a schedule of its own and trigger a retrain the
moment the live data has genuinely shifted.

Method:
- Numeric features (transaction_amount, transaction_hour): two-sample
  Kolmogorov-Smirnov test. This tests whether two samples come from the
  same underlying distribution — exactly the question "has this feature's
  distribution shifted" is asking.
- Categorical feature (is_new_counterparty): two-proportion z-test, same
  approach used in evaluate.py and the A/B Testing Framework project.

A single noisy feature crossing the threshold isn't treated as drift —
requiring at least 2 of 3 features to cross keeps this from false-triggering
on ordinary sampling noise.

Exits non-zero if significant drift is detected, so a CI workflow can use
this to conditionally trigger a retraining run.
"""

import sys
import json
import math
import pandas as pd
from scipy.stats import ks_2samp

BASELINE_SAMPLE_PATH = "artifacts/baseline_sample.csv"
CURRENT_DATA_PATH = "data/transactions.csv"  # TODO: point at live/incoming data in production
DRIFT_REPORT_PATH = "artifacts/drift_report.json"

# p-value below this is treated as "distributions are meaningfully different"
SIGNIFICANCE_THRESHOLD = 0.05

# How many of the checked features must show drift before this is treated
# as real drift rather than noise in one feature.
MIN_FEATURES_FOR_DRIFT = 2


def _proportion_z_test(count1, n1, count2, n2) -> float:
    """Two-proportion z-test, same approach as evaluate.py / A-B testing."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = count1 / n1, count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (p1 - p2) / se


def _p_value_from_z(z: float) -> float:
    """Two-tailed p-value from a z-score, using the error function
    (avoids depending on scipy.stats.norm for this one calculation)."""
    return math.erfc(abs(z) / math.sqrt(2))


def check_numeric_drift(baseline: pd.Series, current: pd.Series, feature_name: str) -> dict:
    statistic, p_value = ks_2samp(baseline, current)
    drifted = bool(p_value < SIGNIFICANCE_THRESHOLD)
    return {
        "feature": feature_name,
        "test": "kolmogorov_smirnov",
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "drifted": drifted,
    }


def check_categorical_drift(baseline: pd.Series, current: pd.Series, feature_name: str) -> dict:
    baseline_true = int(baseline.sum())
    current_true = int(current.sum())
    z = _proportion_z_test(baseline_true, len(baseline), current_true, len(current))
    p_value = _p_value_from_z(z)
    drifted = bool(p_value < SIGNIFICANCE_THRESHOLD)
    return {
        "feature": feature_name,
        "test": "two_proportion_z",
        "baseline_rate": round(baseline_true / len(baseline), 4) if len(baseline) else 0,
        "current_rate": round(current_true / len(current), 4) if len(current) else 0,
        "z_score": round(z, 4),
        "p_value": round(p_value, 4),
        "drifted": drifted,
    }


def main():
    try:
        baseline = pd.read_csv(BASELINE_SAMPLE_PATH)
    except FileNotFoundError:
        print(f"FAIL: no baseline sample found at {BASELINE_SAMPLE_PATH}. Run train.py first.")
        sys.exit(1)

    try:
        current = pd.read_csv(CURRENT_DATA_PATH)
    except FileNotFoundError:
        print(f"FAIL: no current data found at {CURRENT_DATA_PATH}.")
        sys.exit(1)

    print(f"Comparing baseline ({len(baseline)} rows) against current data ({len(current)} rows)...")

    results = [
        check_numeric_drift(baseline["transaction_amount"], current["transaction_amount"], "transaction_amount"),
        check_numeric_drift(baseline["transaction_hour"], current["transaction_hour"], "transaction_hour"),
        check_categorical_drift(
            baseline["is_new_counterparty"].astype(bool),
            current["is_new_counterparty"].astype(bool),
            "is_new_counterparty",
        ),
    ]

    print(f"\n{'Feature':<22}{'Test':<20}{'p-value':<12}{'Drifted?'}")
    n_drifted = 0
    for r in results:
        if r["drifted"]:
            n_drifted += 1
        print(f"{r['feature']:<22}{r['test']:<20}{r['p_value']:<12}{'YES' if r['drifted'] else 'no'}")

    significant_drift = bool(n_drifted >= MIN_FEATURES_FOR_DRIFT)

    report = {
        "significant_drift": significant_drift,
        "n_features_drifted": n_drifted,
        "n_features_checked": len(results),
        "details": results,
    }

    import os
    os.makedirs("artifacts", exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{n_drifted}/{len(results)} features show significant drift.")

    if significant_drift:
        print("DRIFT DETECTED: live data distribution has shifted meaningfully from training data.")
        print("Recommend triggering a retraining run.")
        sys.exit(1)
    else:
        print("No significant drift detected. Model can continue operating on the current schedule.")


if __name__ == "__main__":
    main()