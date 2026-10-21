# IntentOps

**Phase 0 — scaffold only, no trained classifier yet.**

IntentOps is an open-source foundation for developers building local customer-support
intent classification. Future phases will use BANKING77 and a CPU-based classifier;
this phase provides a tested FastAPI service and a small Streamlit status dashboard.

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
must contain 1–5000 characters, and must be a string. Extra fields are rejected.
Valid requests always return HTTP 503 with `error.code: model_not_ready`.
Invalid requests return HTTP 422 without echoing submitted values. There are no
predictions, confidence values, stored messages, or model artifacts.

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
GitHub is not needed for local operation. See `PHASE_0_REPORT.md` for actual
verification results on the implementation machine.

## Layout

| Path | Purpose |
| --- | --- |
| `app/` | API, strict schemas, explicit unavailable inference boundary |
| `dashboard/` | Streamlit interface and local service status |
| `ml/` | Future dataset, training, and evaluation placeholders |
| `tests/` | Self-contained API and inference contract tests |
| `notebooks/` | Valid notebook placeholder for later exploration |
| `artifacts/` | Tracked placeholder; generated binaries ignored |
| `.streamlit/` | Headless, loopback, telemetry-disabled defaults |
| `.github/workflows/` | Optional hosted test automation |

The pre-existing `INTENTOPS_PROJECT_PLAN.md` is preserved as a long-term roadmap.
Its phase numbering differs from this bounded Phase-0 foundation task.

## Licensing and deferred work

Original code is licensed under MIT; see `LICENSE` (IntentOps Contributors).
BANKING77 retains its separate CC BY 4.0 license and attribution obligations;
see `THIRD_PARTY_NOTICES.md`. No dataset is included.

Deferred: BANKING77 processing, leak-free splits, CPU model training, calibration,
confidence thresholds, real predictions, held-out evaluation, and evaluation
dashboards. No model quality claims can be made at this stage.

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
- `model_not_ready` / Model not trained: expected Phase-0 behavior. There is no
  model to download or train yet.
- API unavailable in the dashboard: check `/health`, its configured local URL,
  and API logs, then refresh. The dashboard remains usable without the API.
