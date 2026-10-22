# Changelog

## 1.0.0 — 2026-10-22

First source release of IntentOps, a local CPU application for 77 English banking
intents. Original source is MIT; BANKING77 data and reproduced category metadata
retain separate CC BY 4.0 attribution. No dataset, trained model or policy is bundled.

- Explicit, pinned BANKING77 preparation and fixed TF-IDF/logistic regression
  training with validation-only evaluation and verified local artifact loading.
- FastAPI liveness, readiness, model/policy status, single prediction and ordered
  1–100 message batches with strict validation and an 8 MiB body limit.
- Streamlit single-message and UTF-8/BOM CSV workflows (1 MiB / 200 rows), explicit
  row errors and transient exports without submitted source text or identifiers.
- Optional verified temperature/review-policy sidecar, explicit opt-ins, non-root
  Docker serving, read-only artifact mounts and offline runtime after setup.
- Synthetic regression tests, measured sequential HTTP benchmarks, contribution,
  security, installation and recovery guidance.

Validation accuracy was 84.46% on 2,001 development rows. The validation-derived
601-row policy audit accepted 483 with 38 errors: its one-sided 95% Wilson upper
error bound **10.13% exceeds the 10.00% demonstration budget**. The policy remains
experimental. The official test has not been scored; no final test performance,
production safety, OOD detection or throughput SLA is claimed. Data acquisition
and schema checks in the earlier data phase are distinct from model evaluation.

Release preparation changes packaging/API metadata to 1.0.0 and documentation;
the frozen model, policy, threshold and artifact compatibility checks are unchanged.
