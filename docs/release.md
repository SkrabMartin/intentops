# Installation, release verification and recovery

## Source-only first install

Clone the public repository and select the desired release:

```powershell
git clone https://github.com/SkrabMartin/intentops.git
cd intentops
git checkout v1.0.0
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
```

The tag is available only after release publication. Before publication, use the
reviewed commit explicitly. Linux/macOS use `python3.12 -m venv .venv` and
`source .venv/bin/activate`. No dataset, model or policy ships with the source.
With trust unset, start the [native API/UI](../README.md#native-setup-python-312),
or use the base Docker configuration:

```powershell
$env:INTENTOPS_API_PORT = '18000'
docker compose build
docker compose up -d
Invoke-RestMethod http://127.0.0.1:18000/health
# /health 200; /ready and valid /predict requests 503 model_not_ready.
# Dashboard on http://127.0.0.1:8501 shows model unavailable.
docker compose stop
```

Package/image installation can use the network; startup never downloads data or
fits a classifier. No-model operation is a useful installation check, not inference.

## Explicit fresh-user model and policy bootstrap

These are **future operator setup instructions**, not commands executed against
the protected dataset during v1.0 release verification. The release exercise uses
source-only startup and fictional training/calibration fixtures. Existing trusted
bundles must use restart commands instead; preserve completed audit claims.

1. Read [BANKING77 provenance/licensing](data.md) and
   [third-party notices](../THIRD_PARTY_NOTICES.md). In the installed environment,
   explicitly run `python -m ml.dataset fetch`, `python -m ml.dataset verify`,
   then `python -m ml.dataset prepare --seed 42 --val-fraction 0.20`.
   Fetch downloads **both** official train and reserved test files; verification
   and preparation inspect source bytes. These commands are not safe substitutes
   for a training-only release check. None scores the official test, which remains
   reserved. Offline import requires both pinned original files and the same
   verification; do not bypass hashes or relabel arbitrary fixtures as BANKING77.
2. Native: explicitly run `python -m ml.train` with approved prepared inputs.
   Docker: `docker compose build`, then
   `docker compose -f compose.yaml -f compose.model.yaml --profile setup run --rm --no-deps train`.
   Training reads only original training data and prepared partitions; the original
   test file is not mounted in the training container. It needs no network after
   dependencies and inputs are present. Artifacts are generated in the serving
   environment, never transferred between incompatible Windows/Linux versions.
3. Optional policy, with the base model fixed: native
   `python -m ml.calibrate develop --trust-local-artifacts` then **once only**
   `python -m ml.calibrate audit --trust-local-artifacts`. For Docker, use README's
   [policy setup commands](../README.md#optional-calibration-and-human-review-policy).
   Only approved manifest/validation inputs are needed. The audit is deliberately
   irreversible for that experiment; do not delete a claim or repeat a real audit
   to demonstrate installation. Do not retune the frozen threshold after audit.
4. Start with explicit model trust and optionally policy enablement. Verify
   `/ready` and `/policy/info` separately, then classify fictional examples.
   Exact environment/hash/schema compatibility remains mandatory. See
   [serving](serving.md) for paths, trust and fallback behavior.

The published model metrics use validation data, and the policy audit uses 601
rows derived from that validation. Its 10.13% one-sided 95% upper error bound
exceeds the 10.00% budget. No official held-out score, production safety or OOD
guarantee is established. A frozen PIN-related example can still be wrong at high
score. [Benchmarks](benchmarks.md) describe short sequential runs, not capacity/SLA.

## Ordinary restart and offline proof

With compatible models and policies already generated, run only:

```powershell
$env:INTENTOPS_API_PORT = '18000'
docker compose -f compose.yaml -f compose.model.yaml -f compose.policy.yaml up -d --pull never --no-build
Invoke-RestMethod http://127.0.0.1:18000/ready
Invoke-RestMethod http://127.0.0.1:18000/policy/info
```

Use base + model files for model-only mode (all results need review), base alone
for no-model mode. Restart needs cached images/packages and compatible artifacts,
not datasets, training or downloads. Compose networking itself is not proof of
network isolation. With both default project volumes present, genuinely disable
network access and test a fictional single/batch prediction:

```powershell
docker run --rm --network none --env INTENTOPS_TRUST_LOCAL_ARTIFACTS=1 --env INTENTOPS_ENABLE_REVIEW_POLICY=1 --mount type=volume,source=intentops_models,target=/opt/intentops/artifacts,readonly --mount type=volume,source=intentops_policies,target=/opt/intentops/policies,readonly intentops-api:local python -c "from app.inference import ModelService; s=ModelService.from_environment(); assert s.ready and s.policy_info().status=='ready'; assert len(s.predict_batch(['When will my new card arrive?'] * 100).predictions)==100; print('offline single/batch ready', s.predict('Hello, please help.').intent)"
```

No dataset mounts are needed. Named-volume names assume the default Compose
project `intentops`; use the actual names for another explicitly chosen project.
Native restart likewise uses the existing environment and trust flags, without
rerunning any data/model/policy setup CLI.

## Safe stop, rollback and failed upgrades

```powershell
docker compose -f compose.yaml -f compose.model.yaml -f compose.policy.yaml stop
```

This stops only the named project's services and preserves volumes. Never use
`down -v`, volume/system prune or delete claim markers as a recovery procedure.
Preserve the previous source commit, image identity and original trusted artifact
roots before an upgrade. Check out a previously verified commit in a separate
checkout and run its matching cached environment/image and trusted compatible
bundle. Do not force-reset a checkout with user changes. Stop the current service
before binding its port in the replacement. Recheck liveness, readiness, metadata
and fictional single/batch/CSV results. An older source checkout alone cannot make
an incompatible artifact load. Keep safe no-model mode while resolving mismatch;
never disable compatibility checks. Native services stop with Ctrl+C in their own
terminals. Do not stop unrelated applications occupying port 8000.

## Maintainer release gates

- Preserve history and private reports; inspect tracked/staged inventory and
  Docker context. No data, vocabulary, model, policy, log or private report assets.
- Run full pytest, Ruff lint/format, compileall, whitespace and all three Compose
  config checks. Use synthetic fixtures; no official test evaluation, real refit,
  recalibration, threshold selection or repeat audit as a release gate.
- Independently install source in a new environment and verify honest no-model
  API/UI state. Build final Docker source, check base/model/policy modes, batch100,
  single/CSV export, read-only/non-root execution and actual network-none runtime.
- Compare preserved artifact hashes, inspect sanitized logs, then stop only the
  verification services without deleting volumes.
- Review and normally push the source commit; observe green hosted CI on that
  exact SHA. Verify GitHub's rendered README/tree and licensing/risk links.
- Check remote tag and release: if v1.0.0 exists, stop rather than overwrite it.
  Only after green gates, create an annotated v1.0.0 tag on the verified commit,
  push normally and publish accurate GitHub release notes. Use standard source
  archives only; attach no generated/private files. Verify tag SHA and release.

Version ranges and Docker model constraints are not a full lockfile or an immutable
base image. New environments can change; fail closed on incompatibility. Security
reporting availability is documented honestly in [SECURITY.md](../SECURITY.md).
