# AMLGuard CI/CD Pipeline

A CI/CD pipeline that keeps [AMLGuard AI](link-to-original-repo) — my
fraud/AML detection model — reliable over time, not just accurate on the
day it was trained. It doesn't just train and deploy; it validates,
gates, monitors, and reacts to change on its own.

## Problem

A trained model isn't a finished product. Fraud patterns evolve, data
pipelines change, and a model that scores 0.9914 ROC-AUC today can quietly
degrade over months without anyone noticing — because ML failures are
usually statistical, not crashes. "The training script ran successfully"
does not mean "the resulting model is actually good." This pipeline
exists to close that gap, and to close it automatically.

## What this does

**Core pipeline** (`.github/workflows/ml_pipeline.yml`), on every push and
on a weekly schedule:

1. **Validates incoming data** — schema, value ranges, and null-rate
   checks (Pandera). Bad data never reaches training.
2. **Retrains the model** — XGBoost + Isolation Forest, the same ensemble
   AMLGuard uses in production.
3. **Evaluates against the production baseline** — the new model must
   meet or beat the current baseline (ROC-AUC 0.9914, Recall 0.9361)
   within a small tolerance, or the pipeline fails and production stays
   untouched.
4. **Registers the model** (MLflow) — full lineage: metrics, parameters,
   and the model artifact itself.
5. **Caches a reference sample** of the training data's feature
   distribution, for drift detection to compare against later.

**Drift monitoring** (`.github/workflows/drift-check.yml`), on a daily
schedule:

1. **Compares live/incoming data against the training distribution** —
   a two-sample Kolmogorov-Smirnov test on numeric features
   (`transaction_amount`, `transaction_hour`), and a two-proportion
   z-test on the categorical feature (`is_new_counterparty`).
2. **Flags significant drift** only when at least 2 of 3 features cross
   the significance threshold — one noisy feature isn't treated as real
   drift.
3. **Automatically triggers the core pipeline** (`ml_pipeline.yml`) via
   the GitHub API when drift is detected, instead of waiting on the
   weekly schedule.

## Why this design

- **The evaluation gate is the core of this project.** Without it, a
  pipeline is just automated retraining with no guarantee the result is
  actually deployable. Verified in practice: a deliberately weaker model
  (ROC-AUC 0.76, trained on realistic overlapping synthetic data) was
  correctly blocked from promotion, twice — once manually, once via an
  automatic drift-triggered retrain.
- **Drift detection uses real statistical tests, not eyeballed deltas.**
  A sentiment or feature breakdown wobbles day to day from sampling noise
  alone; the KS test and proportion test only flag a shift when it's
  unlikely to be random noise.
- **Data validation runs first**, because a model trained on corrupted or
  drifted data can look like it trained fine while learning the wrong
  patterns entirely.
- **The two workflows are decoupled but connected** — drift-check doesn't
  retrain anything itself, it only decides *whether* a retrain is
  warranted and hands off to the pipeline that actually knows how to gate
  and register a new model. Each workflow has one job.
- **Fails loudly, not silently** — any stage failing halts the pipeline
  and leaves production untouched, rather than promoting a worse model by
  default.

## Proven, not just built

Both the evaluation gate and the drift monitor have been verified against
real, opposite-outcome test cases — not just written and assumed to work:

- **Evaluation gate:** run against a deliberately weaker model (realistic
  overlapping synthetic data, no more suspiciously perfect scores) —
  correctly failed and blocked promotion. Run against synthetic data
  matching the training distribution — correctly analyzed and reported.
- **Drift monitor:** run against the training data's own distribution —
  correctly reported no drift. Run against a deliberately shifted dataset
  (`generate_drifted_data.py`) — correctly flagged drift on all 3 checked
  features and exited with the signal that triggers a retrain.
- **The full closed loop:** deliberately shifted data was fed into the
  drift monitor, which detected drift and automatically triggered
  `ml_pipeline.yml` with no manual intervention. That retrain ran,
  correctly failed the evaluation gate, and correctly refused to promote
  the resulting model — demonstrating the whole system reacting to a
  change and catching its own retrain's shortcomings, unattended.

## Architecture

```
                    push / weekly schedule
                            │
                            ▼
                   validate_data.py  ──fail──▶ pipeline halted
                            │ pass
                            ▼
                       train.py  (XGBoost + Isolation Forest)
                       + saves baseline_sample.csv
                            │
                            ▼
                     evaluate.py  ──fail──▶ pipeline halted, production untouched
                            │ pass
                            ▼
                  log_to_registry.py  (MLflow — full lineage)


        daily schedule
              │
              ▼
     check_drift.py  ──no drift──▶ done, nothing to do
              │ drift detected
              ▼
   triggers ml_pipeline.yml  ─────────────────▶ (loops back into the flow above)
```

## Run it locally

```bash
git clone <repo-url>
cd amlguard-cicd
pip install -r requirements.txt

# Generate demo data (synthetic, realistic overlapping distributions)
python src/generate_synthetic_data.py

python src/validate_data.py
python src/train.py          # also saves artifacts/baseline_sample.csv
python src/evaluate.py
python src/log_to_registry.py   # requires a running MLflow tracking server, optional

# Drift detection
python src/check_drift.py                    # compares against the same data — expect no drift
python src/generate_drifted_data.py           # creates data/drifted_transactions.csv
# then point CURRENT_DATA_PATH at the drifted file to see drift correctly detected
```

To run the full pipeline the same way GitHub Actions does, see
`.github/workflows/ml_pipeline.yml` and `.github/workflows/drift-check.yml`.

## Roadmap

- Canary / shadow deployment — route a small percentage of real traffic to
  a newly promoted model before fully replacing the production model.
- Only refresh the drift baseline when a model is actually promoted,
  rather than on every training run, so drift is measured against what's
  truly running in production.
- Deployment integration with the AWS SageMaker endpoint explored for
  AMLGuard's original deployment.

## Related

- [AMLGuard AI](link-to-original-repo) — the underlying fraud detection
  model this pipeline trains and validates.
- [AMLGuard Investigation Agent](link-to-agent-repo) — the agentic
  investigation layer built on top of AMLGuard's flagged transactions.
