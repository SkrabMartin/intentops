# IntentOps — Phase 0 Execution Report

## Summary and overall status

**PASS — Phase 0 implementation and verification complete.**
The scaffold is functional and truthful: no model, dataset, predictions, or metrics
are fabricated. Native and Docker smoke tests passed; 22 automated tests pass.
An early provisional report was saved before Docker testing, followed by an
85% checkpoint recording test results and the port conflict. Date: 2026-10-21.

The workspace already contained an initialized Git repository with no commits and
an untracked IntentOps project plan. It was used in place, preserving that plan
and avoiding a nested repository. The requested local initial commit records the scaffold and this report;
no remote repository, publication, push, or cloud resources were created.

## Repository tree created

```text
./
├── app/{__init__.py,api.py,inference.py,schemas.py}
├── ml/{__init__.py,dataset.py,train.py,evaluate.py}
├── dashboard/{__init__.py,streamlit_app.py}
├── tests/{test_api.py,test_inference.py,test_dashboard.py}
├── notebooks/exploration.ipynb
├── artifacts/.gitkeep
├── .github/workflows/tests.yml
├── .streamlit/config.toml
├── .gitignore
├── .dockerignore
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── PHASE_0_REPORT.md
└── INTENTOPS_PROJECT_PLAN.md (preserved)
```

## Files added or modified

All files in the tree except the pre-existing project plan were added (26 files).
The plan's contents were preserved. Local ignored .venv, .uv-python, and .uv-cache
hold the interpreter and installed packages; they are not source files for Git.
The notebook is a valid nbformat-4 JSON document containing an explanatory cell.

## Implementation decisions and rationale

- Python 3.12 is the coherent supported version. Isolated Python 3.12.13 was
  installed because the existing system interpreter is 3.10.6.
- Setuptools packaging; FastAPI, Uvicorn, Streamlit, and HTTPX are the direct runtime
  dependencies. Pytest and Ruff are optional development dependencies. Streamlit
  brings Pandas/NumPy transitively; no ML processing or direct Pandas import exists.
- Dependencies have bounded version ranges, with no full lockfile. Exact future
  dependency reproducibility is not claimed; nothing installs at runtime.
- Strict Pydantic request/response contracts, 1–5000 stripped text characters,
  rejected extra fields and non-string text. Validation responses omit input values.
- ModelNotReadyError produces a stable 503 response. Health is service liveness;
  model readiness is false. No dummy labels or confidence scores exist.
- Dashboard sends only status requests, has disabled prediction, handles an absent
  API, bypasses inherited HTTP proxies, and does not follow redirects.
- Loopback host bindings, uid 10001 in containers, Python-based API health check,
  and one Dockerfile shared by both services. Compose supports host-port overrides.
- Streamlit telemetry is disabled. Startup logs exposed its optional public-IP
  banner lookup; explicit browser.serverAddress now bypasses that lookup, confirmed
  by inspecting the installed Streamlit bootstrap implementation.
- MIT copyright defaults to IntentOps Contributors: no reliable declared project
  copyright owner was present in existing project metadata.

## Commands executed

Executed from C:/Users/skrab/Documents/ChatGPT/datascience-projekt:

```powershell
Get-Location
Get-ChildItem -Force
rg --files -g AGENTS.md -g '!node_modules' -g '!.git'
git status --short --branch
git log -1 --oneline
python --version
py --list-paths
python -m pip --version
uv --version
docker --version
docker compose version
docker info
docker info --format '{{.ServerVersion}}'
git config user.name
git config user.email
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.uv-python'
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv\Scripts\python -m ruff format app ml dashboard tests
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m compileall -q app ml dashboard
docker compose config
docker compose up --build -d
$env:INTENTOPS_API_PORT = '18000'
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose logs --tail=100
git diff --check
docker compose stop
docker compose ps --all
```

Additional short Python probes checked native/Docker HTTP endpoints, parsed the
notebook JSON, and executed Streamlit AppTest inside the dashboard container.
A temporary `docker run --rm -i --network none intentops-api python -` probe
started both services and rendered the dashboard without external networking.
Native probes used subprocesses with hidden windows and stopped only those
processes in finally blocks. Sandbox-hung pytest processes from this workspace
were stopped after the successful run outside the sandbox.

## Automated test results

Final `python -m pytest -q`: **22 passed, 1 warning in 3.67s**.
Tests cover health/model status, valid requests returning 503, text boundaries,
invalid types/extra fields, response schemas rejecting false model claims,
non-echoing validation errors, inference exception and silent output, plus dashboard
rendering with an available and unavailable API. No external services/data required.

Ruff lint: **All checks passed**. Ruff format: **18 files already formatted**.
Compileall: exit 0. Notebook JSON/nbformat check: passed.
Staged whitespace check: new foundation files pass. The preserved plan has four
pre-existing Markdown hard-break lines with trailing spaces (lines 3–6); these
were retained intentionally rather than altering the user's document.

Corrections during verification:
- First executable suite: 19 passed, 1 failed because a test searched for the word
  `input` in human-readable error text (`Extra inputs are not permitted`). Changed
  it to assert the exact allowed JSON keys instead.
- First added dashboard tests: 20 passed, 2 failed with `AppTest script not found`
  because relative paths resolve against the test file. Fixed using an absolute
  path derived from __file__.
- Windows sandbox runs hung inside asyncio's fallback socketpair accept. The same
  self-contained suite passed with local socket access outside the sandbox.
- Remaining warning: StarletteDeprecationWarning, `Using httpx with
  starlette.testclient is deprecated; install httpx2 instead.` HTTPX remains
  supported by the installed client; no failing functionality was observed.

## Docker verification and service URLs

**Verified running** during verification; both project services stopped afterward.
Docker Engine 29.8.0, Compose v5.5.1. Both images built successfully. Compose
reported API healthy and dashboard Up. Host HTTP probes returned:

| URL / action | Actual result |
| --- | --- |
| http://127.0.0.1:18000/health | 200, status ok, model_ready false |
| http://127.0.0.1:18000/model/info | 200, phase 0, untrained, null version |
| http://127.0.0.1:18000/docs | 200, HTML |
| POST http://127.0.0.1:18000/predict | 503, error.code model_not_ready |
| http://127.0.0.1:8501 | 200, HTML |
| http://127.0.0.1:8501/_stcore/health | 200, ok |

Container AppTest rendered the dashboard with actual status data from api:8000
and confirmed the disabled button. Startup/access logs were inspected.
Initial default-port startup failed with `bind: Only one usage of each socket
address ... is normally permitted` on port 8000. Existing unrelated listener was
preserved; override INTENTOPS_API_PORT=18000 resolved it. The default remains 8000.

The network-disabled temporary container returned 200 from both service health
endpoints and rendered the dashboard against the local API. It was removed on
exit. Final rebuilt images passed the same host probes and container dashboard
rendering. Configuration checks confirmed usage statistics false and an explicit
browser address; final logs show only the configured localhost banner.
No volumes or unrelated Docker resources have been removed.

## Native Python verification

Python 3.12.13 installed in the workspace; uv installed 51 packages including
FastAPI 0.158.2, Uvicorn 0.54.0, Streamlit 1.66.1, HTTPX 0.28.1, Pytest 9.1.1,
Ruff 0.17.1, and Pydantic 2.14.1. Editable project installation succeeded.

A native Uvicorn server at http://127.0.0.1:18001 returned 200 for health/model
info and 503 for valid prediction. A native Streamlit server at
http://127.0.0.1:18501 returned 200 for root and /_stcore/health. These alternate
ports avoided existing/project services. Both native processes were stopped.
Dashboard UI behavior also passed real Streamlit AppTest execution.

## Licensing and third-party attribution

LICENSE contains standard MIT text, copyright 2026 IntentOps Contributors.
THIRD_PARTY_NOTICES.md cites the original PolyAI dataset repository, its CC BY 4.0
license, Creative Commons terms, and the Casanueva et al. (2020) paper. Attribution,
license-link, and change-indication obligations are distinguished from the code
license. No BANKING77 contents were downloaded or redistributed. Dependencies
retain their own licenses.

## Offline-first and security checklist

- No paid APIs, API keys, cloud hosting, hosted databases, GPU, or model downloads.
- Both installed services and dashboard rendering tested with Docker network none.
- Usage statistics disabled; explicit browser address disables public-IP lookup.
- Raw message text is not logged, persisted, or sent by the dashboard to the API.
- API validation errors omit submitted values; request access logs omit bodies.
- Native defaults and Docker published ports bind to loopback. Containers non-root.
- Venv, caches, secrets, datasets, checkpoints and model binaries ignored by Git.
- Initial package/interpreter/image downloads require internet unless pre-cached.
- Runtime does not install dependencies or fetch inference artifacts.

## Blockers, limitations, and unresolved issues

No implementation blocker remains. Initial sandbox DNS/daemon/socket restrictions
were resolved by scoped access for setup and verification. Default port 8000 is
occupied on this machine; use the documented override. One upstream test-client
deprecation warning remains. Dependencies/base image are bounded/tagged rather
than fully locked by hashes; a later reproducibility pass may lock them.
No model capability is provided by design. GitHub Actions workflow is present,
but hosted CI was not run because nothing was published. UI was exercised through
Streamlit AppTest and HTTP probes; no manual browser visual inspection is claimed.
Final image smoke checks passed. Project containers were stopped non-destructively.
The initial local commit is the final repository operation; no remote is configured.
The commit includes 27 files (26 new foundation files plus the preserved plan).

## Remaining work before Phase 0 sign-off

No implementation or validation work remains before Phase-0 sign-off. Both
verification containers are stopped; use the commands below to restart them.
The local initial commit records this completed report. All acceptance criteria
were assessed:

| Criterion | Result |
| --- | --- |
| Required core layout | PASS |
| Standard MIT license | PASS |
| Separate dataset attribution/license | PASS |
| Python installation | PASS |
| Truthful API health | PASS |
| Streamlit launches and shows untrained status | PASS |
| Meaningful API/inference tests | PASS |
| CI workflow present | PASS (hosted execution not attempted) |
| Compose configuration and real execution | PASS (port override used) |
| No paid APIs/model downloads/cloud requirements | PASS |
| Native and Docker README workflows | PASS |
| Early report and subsequent updates | PASS |
| No later-phase ML implementation | PASS |

To run after cleanup while port 8000 is busy:

```powershell
$env:INTENTOPS_API_PORT = '18000'
docker compose up -d
```

Then open http://127.0.0.1:18000/docs and http://127.0.0.1:8501.
No additional manual setup is required for the verified scaffold on this machine.

## Handoff instructions for Phase 1

Next action after Phase-0 review: explicitly authorize dataset work. Confirm
BANKING77 source/license/revision, plan checksums and a manual import fallback,
preserve upstream train/test splits, and create seeded training/validation splits
without test leakage. Add data dependencies only then. Training, calibration,
real predictions, thresholds, and evaluation dashboards remain later work.
The preserved long-term plan uses different phase numbering; this report describes
only the bounded foundation prompt executed here.
