"""
validate_data.py

Validates incoming transaction data before it's allowed to reach training.
Bad data should never silently poison a retrained model — this script is
the gate that stops that from happening.

Uses Pandera for declarative schema/value validation. Exits non-zero
(failing the CI/CD pipeline) if any check fails.
"""

import sys
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

DATA_PATH = "data/transactions.csv"  # TODO: point at your real data source

# ---------------------------------------------------------------------------
# Schema definition — adjust column names/ranges to match your real AMLGuard
# feature set. This is a representative example.
# ---------------------------------------------------------------------------

schema = DataFrameSchema(
    {
        "transaction_id": Column(str, nullable=False, unique=True),
        "transaction_amount": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1_000_000),
            ],
            nullable=False,
        ),
        "account_id": Column(str, nullable=False),
        "transaction_hour": Column(
            int,
            checks=Check.in_range(0, 23),
            nullable=False,
        ),
        "is_new_counterparty": Column(bool, nullable=False),
        "label": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
    },
    strict=False,  # allow extra columns without failing
)

NULL_THRESHOLD = 0.05  # fail if any column has more than 5% missing values


def check_null_thresholds(df: pd.DataFrame) -> list[str]:
    """Flag columns whose missing-value rate exceeds NULL_THRESHOLD."""
    problems = []
    null_rates = df.isnull().mean()
    for col, rate in null_rates.items():
        if rate > NULL_THRESHOLD:
            problems.append(f"Column '{col}' has {rate:.1%} missing values (limit: {NULL_THRESHOLD:.0%})")
    return problems


def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"FAIL: data file not found at {DATA_PATH}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows. Running schema validation...")

    try:
        schema.validate(df, lazy=True)
        print("PASS: schema validation")
    except pa.errors.SchemaErrors as err:
        print("FAIL: schema validation")
        print(err.failure_cases)
        sys.exit(1)

    null_problems = check_null_thresholds(df)
    if null_problems:
        print("FAIL: null threshold check")
        for p in null_problems:
            print(f"  - {p}")
        sys.exit(1)
    print("PASS: null threshold check")

    print("All data validation checks passed. Safe to proceed to training.")


if __name__ == "__main__":
    main()