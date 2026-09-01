"""
generate_synthetic_data.py

Creates a synthetic transaction dataset with REALISTIC, OVERLAPPING
fraud/legitimate distributions — not a trivial separable split. Real
fraud detection never hits 100% on every metric; if it does, that's a
signal the data (or the split) is too easy, not that the model is good.

Design choices that create realistic difficulty:
- Fraud and legit transaction amounts come from overlapping gamma
  distributions (fraud skews higher on average, but plenty of legit
  transactions are just as large, and plenty of fraud is small).
- Transaction hour is a soft tendency, not a hard rule — fraud is only
  somewhat more likely at night, with real overlap across all hours.
- is_new_counterparty is probabilistic, not deterministic.
- ~3% label noise is injected — a small fraction of transactions are
  mislabeled, matching real-world label quality issues (a human
  investigator got it wrong, or fraud was never actually confirmed).

This is still synthetic — for demo/pipeline-testing purposes only, not
a substitute for real AMLGuard production data.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 2000
FRAUD_RATE = 0.06

n_fraud = int(N_SAMPLES * FRAUD_RATE)
n_legit = N_SAMPLES - n_fraud

# ---------------------------------------------------------------------------
# Amounts — overlapping distributions. Fraud skews higher on average, but
# there's real overlap: some legit transactions are large, some fraud is
# small.
# ---------------------------------------------------------------------------
legit_amounts = np.random.gamma(shape=2.2, scale=180, size=n_legit)
fraud_amounts = np.random.gamma(shape=2.2, scale=380, size=n_fraud)  # higher mean, same shape -> overlapping spread

# ---------------------------------------------------------------------------
# Hours — soft tendency, not a hard rule. Each class draws from a mixture
# of "normal daytime" and "unusual hours", with fraud weighted more toward
# unusual hours but far from exclusively.
# ---------------------------------------------------------------------------
def sample_hours(n, night_weight):
    hours = np.arange(24)
    # Base daytime bump (7am-9pm) + smaller night presence, blended by night_weight
    day_probs = np.array([1 if 7 <= h <= 21 else 0.3 for h in hours], dtype=float)
    night_probs = np.array([1 if h < 7 or h > 21 else 0.3 for h in hours], dtype=float)
    day_probs /= day_probs.sum()
    night_probs /= night_probs.sum()
    blended = (1 - night_weight) * day_probs + night_weight * night_probs
    blended /= blended.sum()
    return np.random.choice(hours, size=n, p=blended)

legit_hours = sample_hours(n_legit, night_weight=0.15)  # mostly daytime, some night
fraud_hours = sample_hours(n_fraud, night_weight=0.55)  # leans night, but far from exclusive

# ---------------------------------------------------------------------------
# New counterparty — probabilistic tendency, not deterministic
# ---------------------------------------------------------------------------
legit_new_cp = np.random.choice([True, False], size=n_legit, p=[0.15, 0.85])
fraud_new_cp = np.random.choice([True, False], size=n_fraud, p=[0.55, 0.45])

legit = pd.DataFrame({
    "transaction_id": [f"TX-{i:06d}" for i in range(n_legit)],
    "transaction_amount": np.round(legit_amounts, 2),
    "account_id": [f"ACC-{np.random.randint(1000, 1200)}" for _ in range(n_legit)],
    "transaction_hour": legit_hours,
    "is_new_counterparty": legit_new_cp,
    "label": 0,
})

fraud = pd.DataFrame({
    "transaction_id": [f"TX-{i:06d}" for i in range(n_legit, N_SAMPLES)],
    "transaction_amount": np.round(fraud_amounts, 2),
    "account_id": [f"ACC-{np.random.randint(1000, 1200)}" for _ in range(n_fraud)],
    "transaction_hour": fraud_hours,
    "is_new_counterparty": fraud_new_cp,
    "label": 1,
})

df = pd.concat([legit, fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Clip amounts to stay within the schema's allowed range
df["transaction_amount"] = df["transaction_amount"].clip(0, 1_000_000)

# ---------------------------------------------------------------------------
# Label noise — ~3% of labels get flipped, simulating real-world cases
# where the ground truth itself is imperfect (missed fraud, false
# accusations later cleared, etc.). This alone is usually enough to keep
# a model from ever hitting a perfect score.
# ---------------------------------------------------------------------------
NOISE_RATE = 0.03
n_noisy = int(len(df) * NOISE_RATE)
noisy_idx = np.random.choice(df.index, size=n_noisy, replace=False)
df.loc[noisy_idx, "label"] = 1 - df.loc[noisy_idx, "label"]

os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print(f"Generated {len(df)} synthetic transactions -> data/transactions.csv")
print(f"Fraud rate: {df['label'].mean():.2%}")
print(f"Label noise applied to {n_noisy} rows ({NOISE_RATE:.0%})")
print(df.head())