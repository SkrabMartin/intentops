# Model training and validation

Phase 2 adds local CPU model training and validation; Phase 1 remains the data
acquisition/verification layer. API/dashboard serving is deliberately unchanged:
health says model_ready=false and prediction returns 503 model_not_ready. A local
artifact is not a deployed model. No cloud inference or runtime download is used.

## Inputs and experiment protocol

Python 3.12, the existing setuptools/uv workflow, Pandas, scikit-learn, joblib,
threadpoolctl and Matplotlib. Install `python -m pip install -e '.[dev]'` (or
`uv pip install -e '.[dev]'`) once; subsequent training/evaluation are offline.

BANKING77 source/revision/hashes are documented in [data.md](data.md). The exact
Phase-1 split uses 8,002 training and 2,001 validation rows, seed 42, fraction 0.20,
with all 77 original category strings. The original official test set is reserved.
The training loader reuses Phase-1 hashing, CSV parsing, validation, split algorithm
and serialization to verify the manifest, raw training source and canonical
processed membership/bytes. It never calls dataset.verify (which also reads test)
and never opens raw/test.csv. Test identity is referenced from Phase-1 metadata only.
Neither command regenerates inputs or downloads anything.

Only processed training text/labels enter fitting, including the vectorizer.
Validation is used for descriptive model metrics, not fitting. Exactly one
predeclared configuration was used; no parameter sweep or held-out tuning occurred.
Repeated runs of the same configuration are integrity/offline checks, not new trials.

## Architecture and configuration

Baseline: `DummyClassifier(strategy="most_frequent")` fitted using training labels
and a constant numeric input column. It ignores text and predicts the most frequent
training category. Validation predicted-label counts are saved, not hard-coded.

Main estimator: one scikit-learn Pipeline containing `TfidfVectorizer` and
`LogisticRegression`, serialized together:

- Word unigrams and bigrams; lowercase=True, strip_accents=None, token pattern
  `(?u)\b\w\w+\b` (default: word tokens at least two characters).
- min_df=2, max_df=1.0, max_features=60000, sublinear_tf=True, binary=False,
  use_idf=True, smooth_idf=True, norm="l2", float64 features.
- Logistic regression: C=1.0, solver="lbfgs", L2 regularization through the solver's
  default, max_iter=1000, tol=0.0001, fit_intercept=True, class_weight=None,
  random_state=42. No deprecated penalty keyword is passed. Multiclass behavior is
  the installed scikit-learn solver's default; the complete parameters and library
  version are recorded. Native fitted classifier converged in 33 iterations.
- BLAS/OpenMP threadpool limit 1 during fit and timed prediction. CPU-only, no tuning
  parallelism. Convergence warnings abort publication rather than silently passing.

Normalization is a feature transformation only. Source and processed rows, category
case/punctuation and original files remain unchanged. Labels are lexicographically
sorted and checked against saved model classes, probabilities, metrics and matrix.
Full actual pipeline parameters (including defaults), source/configuration/library
identity, model version and artifact SHA-256 values are recorded in metadata.

## Commands and local bundles

```sh
python -m ml.train
python -m ml.evaluate --trust-local-artifacts
```

Defaults are module/repository-relative, regardless of working directory. Both
commands accept --raw-dir, --processed-dir and --artifact-dir. Raw input requires
only train.csv. Use the existing dataset CLI explicitly to recover missing or
invalid data; training fails rather than silently preparing/fetching data.

Training writes an immutable, staged bundle to artifacts/run-<id>/, then atomically
publishes current.json, which identifies the selected run and metadata hash.
A failed staging operation preserves the preceding current run. Old bundles remain
local for review. Bundle names are unique; model-version identity derives from
configuration, verified data identity, ordered labels and exact library versions.
Timings are separated from that reproducibility identity. Files:

- intent_classifier.joblib and most_frequent_baseline.joblib.
- model_metadata.json with source/processed/manifest hashes and full provenance.
- validation_metrics.json: both estimators, differences, all per-class results,
  a complete ordered confusion matrix, top-3 and label-only confusion summaries.
- validation_classification_report.csv and validation_confusion_matrix.png.
- measurements.json: actual fit/batch inference timing, model size and CPU context.

Evaluation recomputes metrics without fitting. It leaves the original training
evidence unchanged and writes validation_recomputed_metrics.json and separate
recomputed measurements inside that run. Versions, data identity and bundle file
checksums must match. Model files are capped at 64 MiB on load. The deserialized
bytes are the same bytes whose hashes were verified. Custom output paths must
also stay out of Git; default data/artifacts directories are ignored.

**Joblib/pickle files may execute code.** The explicit --trust-local-artifacts flag
is an acknowledgement to load only your own trusted output; the default evaluation
command without that flag refuses deserialization. Checksums validate integrity,
not producer authenticity. Never load arbitrary uploaded/downloaded model bundles.
A mismatched/corrupt/incompatible run fails with recovery instructions.

## Measured validation results

Measured 2026-10-21, Python 3.12.13, scikit-learn 1.9.1, Pandas 3.0.6,
NumPy 2.5.4, SciPy 1.18.1; CPU Windows AMD64, 12 logical CPUs, threadpool limit 1.
These are aggregate statistics from a split of official training data under the
separate BANKING77 license, **not final official-test scores**.

| Metric | Most-frequent baseline | TF-IDF + logistic regression |
| --- | ---: | ---: |
| Accuracy | 0.018991 | 0.844578 |
| Macro F1 | 0.000484 | 0.841855 |
| Weighted F1 | 0.000708 | 0.843882 |
| Macro precision | 0.000247 | 0.853573 |
| Macro recall | 0.012987 | 0.840058 |
| Predicted categories | 1 of 77 | 77 of 77 |
| Validation categories with support | 77 | 77 |
| Top-3 accuracy | Not computed | 0.954023 |

Accuracy improvement is 82.5587 percentage points; macro-F1 difference is 0.841370.
Zero-division is defined as 0 for all class/aggregate metrics, including classes the
baseline never predicts. Top-3 uses probability columns in the explicit class order;
ties use stable category order, and tiny test fixtures use min(3, class_count).
Top-3 is distinct from top-1 accuracy and does not define a routing policy.

First native run: 7.121 seconds fitting the main model; compressed artifact
5,265,117 bytes. One complete validation predict_proba batch took 73.126 ms,
amortized 0.0365 ms/message, including vectorization, without warmup. This is a
single local measurement, not single-request API latency or a hardware-independent
guarantee. Repeat runs and different environments may have different timings.
Every run stores its own actual measurements; metadata does not hard-code timings.

## Label-only error analysis

The lowest validation F1 categories included card_acceptance (0.6087, support 12),
virtual_card_not_working (0.6667, support 8), and top_up_failed (0.6923, support 29).
Small supports make category scores unstable; use caution comparing these values.
Five direct_debit_payment_not_recognised rows were predicted as
card_payment_not_recognised; five get_disposable_virtual_card rows as
disposable_card_limits. Other common confusions included bank-transfer vs card
top-up charges and failed vs pending top-ups. These are counted label pairs, not
message excerpts. Overlapping label concepts suggest lexical ambiguity as a
hypothesis for later analysis; no causal claim is established by this matrix alone.

## Limitations, licensing and next phase

English, banking-domain, single-intent benchmark only. Out-of-domain, unfamiliar,
multilingual and multi-intent inputs are not validated. Exact-match duplicate
checks do not establish freedom from semantic leakage. Numerical repeatability
requires matching dependency versions; dependencies are bounded, not hash-locked.
Probabilities have not been calibrated. No abstention threshold, review routing,
held-out test score, production quality or financial deployment guarantee is claimed.

Original project code remains MIT. [Third-party notices](../THIRD_PARTY_NOTICES.md)
retain PolyAI/BANKING77 CC BY 4.0 attribution and Casanueva et al. (2020). Learned
vocabularies and serialized artifacts may contain source-derived terms; they remain
local and are not published. Aggregate validation results are reported with source
attribution. The notebook remains a cleared-output data exploration artifact.

Next phase requires explicit authorization to integrate trusted artifact loading,
startup readiness and real typed predictions into the API/dashboard. Verify
provenance, compatibility, request limits and failure behavior first. Calibration,
threshold design and one-time held-out assessment require explicit later protocols.
