"""
generate_synthetic_data.py

Creates a small synthetic transaction dataset matching the schema expected
by validate_data.py and train.py, so the pipeline can be run and demoed
end-to-end without needing real AMLGuard production data.

Not meant to replace real data — this is purely so the pipeline has
something valid to validate, train on, and evaluate, for demo/portfolio
purposes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 2000
FRAUD_RATE = 0.06  # roughly matches typical imbalanced fraud rates

n_fraud = int(N_SAMPLES * FRAUD_RATE)
n_legit = N_SAMPLES - n_fraud

# --- Legitimate transactions: lower amounts, normal hours, known counterparties
legit = pd.DataFrame({
    "transaction_id": [f"TX-{i:06d}" for i in range(n_legit)],
    "transaction_amount": np.round(np.random.gamma(shape=2.0, scale=150, size=n_legit), 2),
    "account_id": [f"ACC-{np.random.randint(1000, 1200)}" for _ in range(n_legit)],
    "transaction_hour": np.random.choice(range(7, 22), size=n_legit),  # daytime-biased
    "is_new_counterparty": np.random.choice([True, False], size=n_legit, p=[0.1, 0.9]),
    "label": 0,
})

# --- Fraudulent transactions: higher amounts, unusual hours, more new counterparties
fraud = pd.DataFrame({
    "transaction_id": [f"TX-{i:06d}" for i in range(n_legit, N_SAMPLES)],
    "transaction_amount": np.round(np.random.gamma(shape=3.0, scale=800, size=n_fraud), 2),
    "account_id": [f"ACC-{np.random.randint(1000, 1200)}" for _ in range(n_fraud)],
    "transaction_hour": np.random.choice(
        list(range(0, 6)) + list(range(22, 24)), size=n_fraud
    ),  # night-biased
    "is_new_counterparty": np.random.choice([True, False], size=n_fraud, p=[0.7, 0.3]),
    "label": 1,
})

df = pd.concat([legit, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Clip amounts to stay within the schema's allowed range
df["transaction_amount"] = df["transaction_amount"].clip(0, 1_000_000)

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print(f"Generated {len(df)} synthetic transactions -> data/transactions.csv")
print(f"Fraud rate: {df['label'].mean():.2%}")
print(df.head())