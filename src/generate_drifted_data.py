"""
generate_drifted_data.py

Creates a second synthetic dataset that deliberately shifts away from the
original training distribution — simulating what "the live data has
changed" looks like in practice (e.g., a new fraud pattern emerging, or a
genuine shift in customer behavior).

Used to PROVE check_drift.py actually detects drift, rather than just
trusting that it would. Run this, then run check_drift.py against it, and
you should see significant_drift: true.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(123)  # different seed from the original generator, on purpose

N_SAMPLES = 500

# Deliberately shifted: amounts are higher on average, hours skew later,
# and new counterparties are far more common than in the original baseline.
amounts = np.random.gamma(shape=2.2, scale=420, size=N_SAMPLES)  # baseline was scale=180 for legit
hours = np.random.choice(range(14, 24), size=N_SAMPLES)  # baseline leaned daytime 7-21
new_cp = np.random.choice([True, False], size=N_SAMPLES, p=[0.5, 0.5])  # baseline was ~15% for legit

df = pd.DataFrame({
    "transaction_id": [f"TX-D{i:06d}" for i in range(N_SAMPLES)],
    "transaction_amount": np.round(np.clip(amounts, 0, 1_000_000), 2),
    "account_id": [f"ACC-{np.random.randint(1000, 1200)}" for _ in range(N_SAMPLES)],
    "transaction_hour": hours,
    "is_new_counterparty": new_cp,
    "label": 0,  # label doesn't matter for drift detection, only features do
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/drifted_transactions.csv", index=False)

print(f"Generated {len(df)} deliberately-shifted transactions -> data/drifted_transactions.csv")
print("This dataset should trigger drift detection when compared against the training baseline.")
print(df.head())