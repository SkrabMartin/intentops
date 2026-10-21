# IntentOps

**Phase 2 â€” local model training and validation available; web prediction remains disabled.**

IntentOps is an open-source foundation for developers building local customer-support
intent classification. BANKING77 acquisition, strict validation, deterministic
data preparation, and local exploration are implemented. A tested FastAPI service
and Streamlit status dashboard remain available. A CPU TF-IDF/logistic-regression
model can now be trained locally; serving integration is deferred.

No paid APIs, API keys, GPU, hosted database, or external hosting are required.
Internet may be needed for the initial clone, Python installation, dependency
installation, and Docker image downloads. Once installed, local services run
without internet or external inference calls. Dependencies are installed during
setup/build, never at application startup. Current dependency bounds permit patch
and minor updates: this phase does not claim a fully locked reproducible build.

## Native setup (Python 3.12)

From this repository root, create a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
```

If using uv to provision Python instead: `uv venv --python 3.12`, activate as
above, then `uv pip install -e '.[dev]'`. Use one environment consistently.
On Linux/macOS: `python3.12 -m venv .venv` then `source .venv/bin/activate`.

Start each service in its own activated terminal, from the repository root:

```sh
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
python -m streamlit run dashboard/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

- API documentation: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health> (`status: ok`, `model_ready: false`)
- Model status: <http://127.0.0.1:8000/model/info> (`phase: 0`, `status: untrained`)
- Dashboard: <http://127.0.0.1:8501>

`POST /predict` accepts only `{"text": "fictional example"}`. Text is stripped,
must contain 1â€“5000 characters, and must be a string. Extra fields are rejected.
Valid requests always return HTTP 503 with `error.code: model_not_ready`.
Invalid requests return HTTP 422 without echoing submitted values. The web services
do not load local training artifacts, generate predictions, or store messages.

The dashboard defaults to `http://127.0.0.1:8000`. To change it, set
`INTENTOPS_API_URL` (PowerShell: `$env:INTENTOPS_API_URL = 'http://127.0.0.1:8000'`).
Use only trusted local API addresses. It sends status requests only; textarea
contents are never submitted to the API. Streamlit usage statistics are disabled.
An explicit browser address also disables Streamlit's external-IP discovery banner.

## Docker Compose

Install Docker Engine with Compose, or Docker Desktop with a running Linux engine.
Docker Desktop has separate licensing terms; native Python is also supported.
From the repository root:

```sh
docker compose config
docker compose up --build -d
docker compose ps
docker compose logs --tail=100
```

Open the same localhost URLs listed above. Both ports bind to host loopback.
Containers run as an unprivileged user. Inside Compose the dashboard uses
`http://api:8000`; this service hostname is not the native host address.
The API has a Python-based liveness health check; dashboard startup waits for it.
Both services use the same Dockerfile. To stop verification services without
deleting data: `docker compose stop`. No volumes are required.

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
or real-customer performance. Probabilities have not been calibrated; no confidence
threshold or human-routing policy is implemented. **The API/UI still cannot predict.**
See [model methodology, configuration and limitations](docs/model.md).

## Layout

| Path | Purpose |
| --- | --- |
| `app/` | API, strict schemas, explicit unavailable inference boundary |
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

Deferred: web prediction serving, calibration, confidence thresholds, held-out
test evaluation, and evaluation dashboards. The API/dashboard keep their Phase-0
untrained/unavailable contract until serving integration; a local trained artifact
does not change deployed model readiness. Validation results are benchmark-specific.

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
- `model_not_ready` / Model not trained in the dashboard: expected while serving
  is deferred, even after local training succeeds. No artifact is auto-loaded.
- API unavailable in the dashboard: check `/health`, its configured local URL,
  and API logs, then refresh. The dashboard remains usable without the API.
