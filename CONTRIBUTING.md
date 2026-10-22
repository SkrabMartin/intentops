# Contributing

Use GitHub issues for public-safe bugs and proposals and pull requests for changes.
See [security reporting](SECURITY.md) before disclosing a vulnerability. Local
development requires Python 3.12 and no paid API, GPU or hosted service.

From a clone, PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

On Linux/macOS use `python3.12 -m venv .venv` and
`source .venv/bin/activate`, then the same pip/test commands. Tests use fictional
fixtures, not BANKING77 downloads or operator artifacts. Initial package downloads
may need internet; subsequent tests and serving operate locally. Developer
dependencies are bounded ranges, not a universal lockfile. Docker constrains
model-related versions; serialized artifacts require exact compatible environments.

Keep changes focused, describe the problem and resulting behavior, and include
the checks actually run in your pull request. Update relevant docs when changing
contracts. Preserve fail-closed behavior, strict limits, localhost defaults,
disabled telemetry and separate model/policy opt-ins. Add tests for meaningful
behavior changes; avoid unrelated dependency upgrades.

Never submit dataset rows, customer messages, trained models, vocabulary, policies,
audit claims, uploads/exports, screenshots containing private data, raw logs,
internal reports, credentials or machine-specific paths. Inspect `git diff` and
staged filenames. Use invented minimal examples in issues; report OS, Python and
package versions, commit, command, sanitized status/reason codes and expected versus
actual behavior. Checksums do not establish that a joblib producer is trustworthy.

Do not rerun a frozen experiment's once-only audit, retune on its audit rows, or
open/score the reserved official test as part of CI or a routine fix. A future
held-out evaluation needs a separately frozen protocol and explicit authorization.
See [release operations](docs/release.md), [methodology](docs/model.md) and
[license/attribution](THIRD_PARTY_NOTICES.md). Contributions to original project
source use the existing [MIT license](LICENSE).
