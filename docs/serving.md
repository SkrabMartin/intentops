# Trusted local model serving

Serving is CPU-only, local-first and offline after dependency/model setup. A
verified model loads once per process at startup. No dataset, model endpoint,
training, download, hot reload or calibration fitting occurs in requests. Optional
Phase-4 JSON policy activation is documented below.

## Trust and integrity

Only `INTENTOPS_TRUST_LOCAL_ARTIFACTS=1` acknowledges loading your own local
joblib files. Every other value disables deserialization. The artifact root is
operator configuration (`INTENTOPS_ARTIFACT_DIR`, default artifacts/), never an
HTTP parameter. Joblib/pickle can execute code; neither checksums nor this opt-in
authenticate a producer. Do not use arbitrary uploaded/downloaded model bundles.

The existing model_support verifier checks pointer/run paths, symlinks, metadata
hash, approved model configuration, complete file inventory, file size/hash,
reproducibility identity, exact Python/ML versions and class order. It deserializes
the exact checked bytes, then verifies the fitted full TF-IDF/logistic-regression
pipeline and baseline types. Serving retains the main pipeline in memory only.
An unexpected extra file, incompatible bundle or missing file fails closed.

For dataset-free runtime, serving_contract pins the already-verified Phase-1
input hashes, manifest identity, 77 labels, 8,002/2,001 membership counts, seed 42
and fraction 0.20. It feeds that independently approved identity into the same
provenance-equality checks used by evaluation; it does not trust provenance copied
from the candidate bundle. Raw/processed datasets are never opened by serving.
A genuinely different preparation identity requires explicit review and a contract
update, not weaker checks or blindly copying candidate metadata.

Missing/untrusted/invalid artifacts leave the process alive but not ready. Public
and operator errors contain safe reason codes, not request values, exception text,
paths or secrets. Severe prediction errors disable readiness until restart. There
is no automatic pointer reload; replace bundles via a controlled service restart.

## Native PowerShell

Use the existing Python 3.12 environment and native compatible Phase-2 bundle. For
a new environment install the project and explicitly train with verified local
inputs first; do not download anything on service startup.

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
# Only if a compatible local bundle is absent:
python -m ml.train
$env:INTENTOPS_TRUST_LOCAL_ARTIFACTS = '1'
$env:INTENTOPS_ARTIFACT_DIR = (Resolve-Path artifacts).Path
python -m uvicorn app.api:app --host 127.0.0.1 --port 18000 --no-access-log
```

Another activated terminal:

```powershell
$env:INTENTOPS_API_URL = 'http://127.0.0.1:18000'
python -m streamlit run dashboard/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Only train when deliberately producing/replacing an artifact; normal service use
starts at the trust/API steps. To test no-model mode, set trust to '0' and restart.
Exact environment compatibility is mandatory; regenerating in the serving
environment is the remedy for version mismatch. Do not suppress version checks.

## Docker Compose: same-image setup, explicit opt-in

Base Compose starts safely without a model or trust. For genuine serving, build,
explicitly generate the model in the API image, then opt into the model overlay:

```powershell
$env:INTENTOPS_API_PORT = '18000'
docker compose build
docker compose -f compose.yaml -f compose.model.yaml --profile setup run --rm --no-deps train
docker compose -f compose.yaml -f compose.model.yaml up -d
Invoke-RestMethod http://127.0.0.1:18000/ready
```

Both API and training use the locally built intentops-api:local image. Training
mounts only raw/train.csv and processed/ read-only (never raw/test.csv), and writes
the complete bundle into project volume intentops_models. The Dockerfile seeds an
empty artifact directory owned by uid 10001, so volume initialization permits the
same non-root user to write. Serving mounts this volume read-only and sets explicit
trust only in compose.model.yaml. No model/data is copied into images or Git.
The setup profile is excluded from ordinary up; no automatic training occurs.

Optional permissions check (uid should be 10001 and access True):

```powershell
docker compose -f compose.yaml -f compose.model.yaml --profile setup run --rm --no-deps train python -c "import os; print(os.getuid(), os.access('/opt/intentops/artifacts', os.W_OK))"
```

Windows Python 3.12.13 and container Python 3.12.15 artifacts have different
identities. Keep the native artifact local; produce the Linux bundle using the
API image rather than force-loading Windows files. Rebuilds may update dependencies;
explicitly retrain and restart using that same image. Native setup remains an option
where Docker Engine/Compose or separately licensed Docker Desktop is unavailable.

```powershell
docker compose -f compose.yaml -f compose.model.yaml ps
docker compose -f compose.yaml -f compose.model.yaml logs --tail=100
docker compose -f compose.yaml -f compose.model.yaml stop
```

`stop` preserves the volume. Do not delete user data or use global prune. To return
to safe no-model mode, start with `docker compose up -d` using the base file only.
The API then has no trust setting or model volume. The API liveness health check
allows the dashboard to start even when no classifier is ready.

## API contracts and local examples

| Endpoint | Ready | Not ready |
| --- | --- | --- |
| GET /health | 200, model_ready=true | 200, model_ready=false |
| GET /ready | 200, loaded model metadata | 503, error.code=model_not_ready |
| GET /model/info | 200, phase 4, loaded, version/type/category count | 200, unavailable, null version/type, count 0, safe reason |
| POST /predict | 200, typed real prediction | 503 model_not_ready |
| GET /docs | OpenAPI documents the actual schemas | Available |

Invalid request types, extra fields, blank/overlong input return 422 without
submitted values. The text limit stays 5,000 trimmed characters. An internal
inference failure returns sanitized 500 inference_failed and disables readiness.

Executable request/response example, using fictional text, without embedding a
fabricated or published score transcript:

```powershell
$body = @{ text = 'When will my new card arrive?' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:18000/predict -ContentType 'application/json' -Body $body
```

```python
import httpx
from app.schemas import PredictionResponse

with httpx.Client(trust_env=False, follow_redirects=False) as client:
    response = client.post(
        "http://127.0.0.1:18000/predict", json={"text": "When will my new card arrive?"}
    )
    response.raise_for_status()
    prediction = PredictionResponse.model_validate(response.json())
    assert prediction.intent == prediction.top_predictions[0].intent
    assert len(prediction.top_predictions) == 3
    assert prediction.score_type in {
        "uncalibrated_model_probability",
        "temperature_scaled_probability",
    }
    print(prediction.model_dump())  # Actual output from your loaded model.
```

Responses contain intent, model_version, three distinct intent/score entries and
score_type. Scores are finite in [0,1], descending, with stable declared-class-order
ties. Their sum need not equal one because only the top three are returned. These
are raw without a policy or temperature-scaled with a verified policy. Neither
is a correctness guarantee or OOD detector; recommendations execute no action. Validation-only
performance is in [model.md](model.md); no official held-out score is claimed.

## UI, privacy and diagnostics

Open http://127.0.0.1:8501. The UI separates unreachable API, online/unavailable
model, and ready model. Classify enables only with ready status and valid text;
it sends that text to the configured local API only when clicked. Actual intent,
version and three scores are shown. Timeouts, invalid responses and HTTP failures
produce sanitized messages rather than stale/fake results.

INTENTOPS_API_URL permits loopback HTTP addresses or Compose host api only;
redirects/proxy inheritance are disabled. Request text is ephemeral in the current
browser session and is not saved to disk, databases, logs or telemetry. Native
instructions and Docker disable access logs. Existing Streamlit telemetry/public-IP
lookup settings remain. No CSS/CDN or external model-provider requests are added.

If /ready is 503, inspect /model/info reason: disabled means the exact trust opt-in
was absent; invalid_artifacts means bundle/path/hash/config/version checks failed;
internal_error means inference failed and requires inspection/restart. Inspect your
local bundle with the trusted evaluation CLI in the correct environment, never by
relaxing checks. Other applications on port 8000 are preserved; use host override
18000, and keep the Docker-internal api:8000 address unchanged.

Phase 4 implements optional calibration and development-policy recommendations.
No business routing, held-out evaluation, new base algorithm or production SLA
is implemented here. Original code stays MIT; BANKING77/derived
metadata keep the separate CC BY 4.0 attribution in THIRD_PARTY_NOTICES.md.


## Optional Phase-4 JSON sidecar

Run README's exact `ml.calibrate develop` then `ml.calibrate audit` commands in the
existing model environment. No base retraining. The CLI reads only pinned prepared
manifest.json and validation.csv; original test contents never enter. Options
--processed-dir, --artifact-dir, --policy-dir select operator-local directories;
policy storage must be separate from model/data input locations. Defaults are
module-relative. No downloads occur.

Ignored policies/ holds draft.json and unique draft-<uuid>/record.json containing
the predeclared protocol, T, model/provenance/class-order/software binding, split
IDs/supports/hashes, fitting/selection metrics, full curve and selected rule.
Audit claims the draft once and publishes policy-<uuid>/record.json with unchanged
frozen development and independent audit metrics. current.json selects that record
with SHA-256. Policy version is review- plus its digest prefix. Original model runs
and model current pointer remain unchanged.

JSON is non-executable, deterministically serialized, no NaN/Infinity, duplicate
keys or unknown schema fields. Reads are bounded to 2 MiB per file. Selected policy
runs permit only record.json. Symlinks, traversal, unexpected files, invalid
statistics/schema, unsupported method/version, model/data/class-order/environment
mismatch and bad hashes fail closed. Publication stages a run, renames it, then
atomically replaces the pointer last. Failed pointer publication preserves the
previous policy; orphan runs are not activated. Hashes are not signatures or author
authentication: enable only your own reviewed experiment outputs.

Native activation before API startup:

```powershell
$env:INTENTOPS_TRUST_LOCAL_ARTIFACTS = '1'
$env:INTENTOPS_ENABLE_REVIEW_POLICY = '1'
$env:INTENTOPS_POLICY_DIR = (Resolve-Path policies).Path
python -m uvicorn app.api:app --host 127.0.0.1 --port 18000 --no-access-log
```

Load once at startup; restart to activate a different operator-selected policy.
Only exact '1' enables it; executable joblib trust remains separate. HTTP requests
cannot select paths, thresholds, temperature or reload behavior.
Model ready with disabled/missing/invalid policy keeps /ready 200, /policy/info
reports disabled/unavailable/invalid, and raw predictions recommend needs_review
with reason=policy_not_available. No model means readiness/prediction 503.
/health remains process liveness.

With verified policy, /predict retains intent/model_version/top_predictions and
returns score_type=temperature_scaled_probability, policy_status=ready,
policy_version, policy_threshold, recommendation and reason. Threshold equality
accepts. Null review-all always needs_review. Otherwise reasons are
below_configured_threshold or meets_configured_threshold. The schema rejects
automatic suggestions without a ready policy. Nothing routes, sends, approves or
rejects tickets. Top-three scores are never renormalized.

/policy/info reports non-sensitive status, version, method, temperature, threshold,
mode, demo budget, audit counts and upper error bound. Ready is an integrity state,
not a safety certificate; see [calibration.md](calibration.md).

Docker uses the three Compose files in README. The setup-profile calibrate service
mounts models read-only, only manifest/validation inputs read-only, and writable
project-scoped intentops_policies. Serving mounts both model and policy volumes
read-only under UID 10001. Windows policies must not be force-loaded with a Linux
model identity. Preserve existing models; explicitly generate the compatible
sidecar in the serving image. Ordinary up never trains, calibrates or audits.
Base+model alone disables policy; base alone preserves no-model operation.

After setup, a real offline prediction with both volumes read-only and no inputs:

```powershell
docker run --rm --network none --env INTENTOPS_TRUST_LOCAL_ARTIFACTS=1 --env INTENTOPS_ENABLE_REVIEW_POLICY=1 --mount type=volume,source=intentops_models,target=/opt/intentops/artifacts,readonly --mount type=volume,source=intentops_policies,target=/opt/intentops/policies,readonly intentops-api:local python -c "from app.inference import ModelService; s=ModelService.from_environment(); assert s.ready and s.policy_info().status=='ready'; print(s.predict('When will my new card arrive?').model_dump())"
```

Ordinary Compose networking differs from --network none. Stop with the same
three Compose files and stop; preserve volumes and audit claims. Policy outputs,
plots, logs, screenshots, local verification scripts/results and reports stay
ignored and excluded from builds. No persistent customer-text history exists.


Docker pins model-related packages in docker-constraints.txt to preserve the
existing Phase-3 model environment (pandas 3.0.6, sklearn 1.9.1, NumPy 2.5.4,
SciPy 1.18.1, joblib 1.6.0). This is not a complete dependency lock. Exact Python
and package checks still apply; an incompatible base image fails closed.
