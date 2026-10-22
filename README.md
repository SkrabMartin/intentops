# IntentOps

**Phase 4 — local calibration and human-review recommendations, enabled explicitly.**

IntentOps is an open-source, CPU-only customer-support intent classifier using
BANKING77 and a TF-IDF/logistic-regression pipeline. It includes verified data
preparation, local training/validation, FastAPI predictions and a Streamlit interface.
No API keys, paid model services, GPU, hosted database or mandatory cloud hosting.

Initial dependency/image/dataset setup may need internet. Once installed and a
compatible model is available, inference runs locally without external calls.
Models, learned vocabularies, datasets and internal reports stay outside Git and
Docker build contexts. Training/downloads never happen on application startup.
Dependencies are bounded, not fully locked; exact model environment checks apply.

## Native setup (Python 3.12)

From the repository root (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

Use the existing environment if already set up. With uv: `uv pip install -e '.[dev]'`.
Linux/macOS can use `python3.12 -m venv .venv` and `source .venv/bin/activate`.
A trusted compatible model must exist under artifacts/; if absent, explicitly
prepare verified local data and run `python -m ml.train` as documented below.
No data is required during serving; the approved Phase-1 input identity is pinned.

Start the API with deliberate trust acknowledgement, in an activated terminal:

```powershell
$env:INTENTOPS_TRUST_LOCAL_ARTIFACTS = '1'
$env:INTENTOPS_ARTIFACT_DIR = (Resolve-Path artifacts).Path
python -m uvicorn app.api:app --host 127.0.0.1 --port 18000 --no-access-log
```

In another activated terminal:

```powershell
$env:INTENTOPS_API_URL = 'http://127.0.0.1:18000'
python -m streamlit run dashboard/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

- API docs: <http://127.0.0.1:18000/docs>
- Liveness: <http://127.0.0.1:18000/health> (200 even without a model).
- Readiness: <http://127.0.0.1:18000/ready> (200 only with a verified loaded model).
- Metadata: <http://127.0.0.1:18000/model/info>.
- Dashboard: <http://127.0.0.1:8501>.

For safe no-model mode, remove the trust variable or set it to `'0'` before
starting/restarting the API. Missing/corrupt/incompatible/untrusted artifacts keep
it alive but not ready; /predict returns 503 model_not_ready. Only the exact value
`1` enables loading. Joblib/pickle can execute code: trust only your own artifacts.
This opt-in and SHA-256 checks do not authenticate an artifact producer.

POST /predict accepts only `{"text": "fictional example"}` with a trimmed,
nonblank string of at most 5,000 characters. It returns intent, model_version,
three ranked intent/score pairs and score_type=uncalibrated_model_probability.
With no active policy, every prediction recommends human review. Policy-enabled
responses use temperature-scaled scores and a development threshold. Neither mode
executes routing or establishes a correctness guarantee.
Invalid requests return 422 without echoing submitted values. The UI enables
Classify only when the API reports a ready model and valid input, and submits
text only on click to its configured local API. Messages are ephemeral, not logged
or stored. Usage telemetry and public-IP lookup remain disabled.

## Docker: explicit compatible-model setup and serving

A Windows Python 3.12.13 model must not be force-loaded in Linux Python 3.12.15.
Build the API image and run the existing training command in that same environment.
Only original train.csv and prepared partitions/manifest are mounted read-only;
raw/test.csv is not exposed. A project-scoped named volume preserves the generated
Linux bundle. The serving API mounts that volume read-only as non-root uid 10001.
The image seeds only an empty, owned artifact directory, never model contents.

With local Phase-1 training inputs present, run in PowerShell:

```powershell
$env:INTENTOPS_API_PORT = '18000'
docker compose build
docker compose -f compose.yaml -f compose.model.yaml --profile setup run --rm --no-deps train
docker compose -f compose.yaml -f compose.model.yaml up -d
Invoke-RestMethod http://127.0.0.1:18000/ready
```

The optional compose.model.yaml is explicit trust opt-in for your own local bundle.
Its training service runs only via the setup profile and explicit run command;
ordinary up does not train. Both training and API use intentops-api:local. After
rebuilding/changing the environment, explicitly regenerate a compatible bundle
and restart; version checks are never suppressed. All source inputs stay local.

```powershell
docker compose -f compose.yaml -f compose.model.yaml ps
docker compose -f compose.yaml -f compose.model.yaml logs --tail=100
docker compose -f compose.yaml -f compose.model.yaml stop
```

Open the same localhost URLs listed above. Both published ports bind to loopback;
inside Compose the dashboard uses http://api:8000. The API health check is liveness;
use /ready for model readiness. No volumes are deleted by stop. Data and model
volumes are not included in Git or images. Native Python remains an alternative
to Docker Engine/Compose or separately licensed Docker Desktop.

For default **no-model** startup: `docker compose up -d` (base file only). This
mounts no model and sets no trust opt-in; health is 200, readiness/prediction 503.
See [serving operations and troubleshooting](docs/serving.md) for the exact trust,
integrity, offline and API contracts.

## Optional calibration and human-review policy

The base classifier remains fixed. Phase 4 partitions the already-used validation
set into **800 calibration / 600 threshold-selection / 601 internal-audit rows**.
Temperature scaling preserves the top category. Scores and recommendations are
benchmark-specific, not correctness guarantees or out-of-domain detection.

Native, activated matching environment (run these two steps once, in order):

```powershell
python -m ml.calibrate develop --trust-local-artifacts
python -m ml.calibrate audit --trust-local-artifacts
$env:INTENTOPS_TRUST_LOCAL_ARTIFACTS = '1'
$env:INTENTOPS_ENABLE_REVIEW_POLICY = '1'
python -m uvicorn app.api:app --host 127.0.0.1 --port 18000 --no-access-log
```

Start Streamlit in another terminal using the native commands above. Existing
frozen experiments refuse redevelopment and repeat audit. On this machine the
native policy already exists: use only the environment variables and startup.
Disable policy with `INTENTOPS_ENABLE_REVIEW_POLICY='0'` and restart; model-only
predictions remain available and always recommend review. `/ready` means model
ready; `/policy/info` separately reports policy state and internal audit summary.

Docker, after the compatible Phase-3 model exists (do not retrain that model):

```powershell
$env:INTENTOPS_API_PORT = '18000'
docker compose build
docker compose -f compose.yaml -f compose.model.yaml -f compose.policy.yaml --profile setup run --rm --no-deps calibrate
docker compose -f compose.yaml -f compose.model.yaml -f compose.policy.yaml --profile setup run --rm --no-deps calibrate python -m ml.calibrate audit --processed-dir /input/processed --trust-local-artifacts
docker compose -f compose.yaml -f compose.model.yaml -f compose.policy.yaml up -d
Invoke-RestMethod http://127.0.0.1:18000/policy/info
```

Setup mounts only manifest.json and validation.csv read-only; no official test or
training text is needed. The unchanged model volume and separate policy volume
are read-only during serving. Ordinary startup never fits/calibrates/audits.
For restart use only the `up -d` command; stop with the same three `-f` files and
`stop`, preserving volumes. Use just base + model files for policy-disabled mode,
or base alone for safe no-model mode. Dashboard: http://127.0.0.1:8501;
API docs: http://127.0.0.1:18000/docs.

Observed native temperature **0.344899**, selected threshold **0.66**. Audit coverage
was **80.37%**, accepted-item error **7.87%**, review rate **19.63%**. The one-sided
95% Wilson upper error bound was **10.13%**, slightly above the chosen 10% demo
budget. The frozen policy was preserved without retuning. This is an internal
audit of previously evaluated validation data, not an untouched official test.
See [method, formulas, limitations and aggregate results](docs/calibration.md) and
[policy storage/operations](docs/serving.md). Generated policies and reports stay local.

## Development and tests

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Tests use an in-process client and require no server, dataset, network, or GPU.
The optional GitHub Actions workflow runs these checks on push/pull request;
GitHub is not needed for local operation.

## BANKING77 data preparation

After activating the installed environment, run these commands from the root
(identical in PowerShell, Linux, and macOS shells):

```sh
python -m ml.dataset fetch
python -m ml.dataset verify
python -m ml.dataset prepare --seed 42 --val-fraction 0.20
python -m ml.dataset summary
```

Only `fetch` uses the internet. It intentionally acquires the official PolyAI
CSVs at a pinned revision and checks approved SHA-256 hashes before publishing
files into `data/raw/`. Existing mismatches are preserved and cause failure;
inspect and move a mismatched file aside manually before retrying. Downloads use
temporary files, bounded transfers, and finite timeouts, with no fallback hosts.
Atomic raw-file publication requires a filesystem supporting hard links (such as
NTFS or a standard Linux filesystem); unsupported filesystems fail safely.

For **manual offline import**, copy the original `train.csv` and `test.csv` into
`data/raw/`, then run `verify`, `prepare`, and `summary`. No network is required.
Every command accepts `--raw-dir PATH`; preparation also accepts `--output-dir PATH`.
Defaults are relative to the installed project modules, independent of the shell's
current directory. Use separate, non-nested raw and processed locations.

`verify` checks file hashes, UTF-8 CSV schema, nonempty text/category fields, exact
counts and category inventory, then prints JSON with aggregate quality statistics.
`summary` prints the same verified source EDA without creating files. `prepare`
writes `data/processed/train.csv`, `validation.csv`, and `manifest.json` and prints
the manifest. The manifest records source revision/hashes, category frequencies,
row membership, configuration, library version, output hashes and overlap checks.
No timestamps or machine-specific absolute paths appear in deterministic outputs.
On PowerShell, save console JSON with `... > data/summary.json` if desired.

Original CSVs use string `category`, preserving case and punctuation, including
`Refund_not_showing_up` and `reverted_card_payment?`. Hugging Face's numeric `label`
interface is different. The official 10,003-row training set alone is stratified
into **8,002 train / 2,001 validation** rows with seed 42 and fraction 0.20.
All 3,080 official test rows remain unchanged and reserved for later final assessment.
Source IDs are zero-based data-record indices. Fraction must be finite and strictly
between 0 and 1; seed is an integer from 0 through 4294967295. Each category must
fit in both partitions; impossible settings fail clearly.

Rows are neither normalized nor deduplicated. Exact-text overlaps are reported;
this does not detect paraphrases or guarantee freedom from semantic leakage.
Repeatability is byte-for-byte for the same source/configuration/library versions;
dependencies are bounded rather than fully locked. Raw/derived data stay Git-ignored
and excluded from Docker build contexts. Keep custom output paths out of Git too.
See [data provenance and limitations](docs/data.md) for verified hashes and details.

For the local EDA notebook (no automatic downloads or package installation):

```sh
python -m pip install -e '.[dev,notebook]'
python -m jupyterlab notebooks/exploration.ipynb
```

With uv, use `uv pip install -e '.[dev,notebook]'`. Select this project's Python
3.12 environment as the notebook kernel. Missing data produces setup instructions.
The notebook displays frequencies, lengths, imbalance and duplicate statistics,
without raw messages or model scores. Committed outputs are cleared.

## CPU model training and validation

Activate the installed Python 3.12 environment and prepare local data first.
Training and evaluation never fetch data or open the official test file:

```sh
python -m ml.train --help
python -m ml.train
python -m ml.evaluate --help
python -m ml.evaluate --trust-local-artifacts
```

`train` verifies the Phase-1 manifest, original training identity, exact partition
bytes and membership (8,002 train / 2,001 validation, seed 42, fraction 0.20).
It fits a most-frequent DummyClassifier and one word-(1,2) TF-IDF + logistic
regression pipeline on training only, then computes validation-only metrics.
No vocabulary is fitted on validation; the 3,080 official test examples stay reserved.
Missing or modified data fail cleanly; no files are silently regenerated.

Each successful run stages a complete bundle into `artifacts/run-<id>/` and publishes
`artifacts/current.json` last. Older runs are retained. Bundles contain the complete
pipeline, baseline, metadata/configuration/hashes, metrics JSON, per-class CSV,
77×77 confusion matrix PNG, and separate local timing measurements. Everything
under artifacts/ except .gitkeep stays ignored and excluded from Docker images.

`evaluate` requires explicit trust acknowledgement and recomputes validation metrics
from the saved bundle without retraining. It validates metadata, versions, data
provenance and file hashes before deserializing the exact checked bytes. **Joblib
uses pickle and can execute code: only load your own trusted training output.**
Checksums detect corruption; they cannot establish the trustworthiness of a producer.
Recomputed metrics/timings are separate files; original evidence is retained.

Both CLIs accept `--raw-dir`, `--processed-dir`, and `--artifact-dir`; defaults are
module-relative, independent of shell working directory. Keep custom artifact
locations out of Git. `--raw-dir` needs only original train.csv: test.csv is never
opened in training or evaluation. Manifest test hashes are provenance references only.
Use identical dependency versions to reproduce numerical results.

Measured on 2026-10-21 with Python 3.12.13 / scikit-learn 1.9.1, validation only:

| Model | Accuracy | Macro F1 | Weighted F1 | Top-3 accuracy |
| --- | ---: | ---: | ---: | ---: |
| Most-frequent baseline | 1.8991% | 0.000484 | 0.000708 | Not computed |
| TF-IDF + logistic regression | 84.4578% | 0.841855 | 0.843882 | 95.4023% |

All 77 labels have validation support; the main model predicts all 77, baseline one.
Top-3 counts the correct label among three candidates; it is not top-1 routing
accuracy. These BANKING77-derived validation aggregates do not establish held-out
or real-customer performance. Optional Phase-4 calibration and review recommendations
use a separate internal validation protocol; see [calibration results](docs/calibration.md). **API/UI inference requires an explicitly trusted, verified loaded model.**
See [model methodology, configuration and limitations](docs/model.md).

## Layout

| Path | Purpose |
| --- | --- |
| `app/` | API, strict schemas, one-time trusted model loading and real inference |
| `dashboard/` | Streamlit interface and local service status |
| `ml/` | Verified data preparation, CPU training, validation and artifact integrity |
| `tests/` | Self-contained API and inference contract tests |
| `notebooks/` | Runnable, local-only BANKING77 exploration |
| `data/` | Ignored local raw/processed data; never bundled in images |
| `docs/` | Dataset provenance, preparation policy and limitations |
| `artifacts/` | Ignored local models, provenance, validation reports and plots |
| `.streamlit/` | Headless, loopback, telemetry-disabled defaults |
| `.github/workflows/` | Optional hosted test automation |

## Licensing and deferred work

Original code is licensed under MIT; see `LICENSE` (IntentOps Contributors).
BANKING77 retains its separate CC BY 4.0 license and attribution obligations;
see `THIRD_PARTY_NOTICES.md`. No dataset is included.

Deferred: bounded batch/CSV work, performance polish and separately authorized
held-out test evaluation. No external ticket routing is implemented. Validation results remain
benchmark-specific; real serving does not establish production quality.

## Troubleshooting

- Missing Docker or inaccessible daemon: start the Linux engine, check `docker info`,
  or use native Python. Do not infer readiness from configuration alone.
- Port 8000/8501 busy: stop your conflicting service or select other host ports.
  For native API changes, also update `INTENTOPS_API_URL`. For Compose, change
  host-side ports only; keep the internal `api:8000` address.
  Compose also accepts `INTENTOPS_API_PORT` and `INTENTOPS_DASHBOARD_PORT`;
  for example, `$env:INTENTOPS_API_PORT = '18000'` selects host port 18000.
- Missing dependencies: activate `.venv` and repeat the development installation.
  Python 3.12 is required; Python 3.10 is not supported by this project.
- `model_not_ready`: check explicit trust, artifact permissions, complete bundle
  and exact Python/package versions. See the sanitized reason in /model/info.
  Never force-load an incompatible bundle; regenerate in the serving environment.
- API unavailable in the dashboard: check `/health`, its configured local URL,
  and API logs, then refresh. The dashboard remains usable without the API.


Docker pins model-related packages in docker-constraints.txt to preserve the
existing Phase-3 model environment (pandas 3.0.6, sklearn 1.9.1, NumPy 2.5.4,
SciPy 1.18.1, joblib 1.6.0). This is not a complete dependency lock. Exact Python
and package checks still apply; an incompatible base image fails closed.
