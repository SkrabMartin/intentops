# Security policy

The current source release is v1.0.0. Security fixes are considered on the current
default branch; no response deadline, maintenance SLA or enterprise certification
is promised. This application is intended for a trusted local operator, not
untrusted multi-user or public internet exposure. It has no authentication,
tenant isolation or business authorization layer. Keep ports on loopback and do
not process sensitive banking/customer information in public demonstrations.

## Reporting

As verified on 2026-10-22, this repository's GitHub private vulnerability reporting
is **disabled**. There is no verified private project contact published here.
Do not assume that a private report or email channel exists.

For a public-safe concern, open a minimal, sanitized
[GitHub issue](https://github.com/SkrabMartin/intentops/issues/new).
For a sensitive finding, use an issue only to request that the maintainer enable
a private disclosure channel; omit vulnerability details, exploit payloads, customer
text, logs, secrets and attachments until a private channel is confirmed. This
limitation can delay confidential handling. GitHub account/platform incidents
belong with GitHub support, rather than public project issues.

## Boundaries and limitations

- Joblib/pickle loading can execute code. Only acknowledge trust for your own
  generated artifacts. Hashes and schema checks detect corruption; they do not
  authenticate a producer. Exact Python/package checks must remain enabled.
- Serving loads verified artifacts once. Model/policy volumes are read-only in
  serving containers, which run as UID 10001. Training/setup is separate and
  explicitly writable. Invalid model state disables predictions; an inactive
  policy falls back to raw scores and human review.
- Local HTTP clients reject nonlocal destinations, proxies and redirects.
  Streamlit telemetry is disabled; request messages and CSV contents are transient
  and omitted from application logs, errors and exported prediction metadata.
  Operator terminals, browser state and third-party tooling still require care.
- Batch/CSV limits bound individual requests, not global concurrent demand. This
  is not a hardened denial-of-service boundary or a verified OOD detector.
- Review recommendations execute no ticket dispatch or financial action. The
  internal audit's 10.13% upper error bound exceeded its 10.00% budget. A high
  probability does not establish correctness, including for PIN-related messages.

Dependencies retain their own licenses and security lifecycles. Version ranges
and model constraints are not a full lockfile. Review updates and reverify a
compatible environment; never weaken artifact checks to accept an update.
See [serving](docs/serving.md) and [safe stop/recovery](docs/release.md).
