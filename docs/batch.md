# Bounded batch and private CSV workflow

## JSON API

POST /predict/batch accepts exactly an object with messages: 1–100 strings,
stripped exactly as single /predict, 1–5000 characters each. Missing/extra fields,
invalid elements, empty or 101-element lists reject the whole request with 422.
No row is silently discarded. Response predictions is an ordered list of existing
PredictionResponse records: intent, top_predictions (3), model_version, score_type,
recommendation, policy_status/version/threshold and safe reason. No input echo.
Single and batch use the same in-memory model/policy path and stable class-order
ties; probability comparisons allow absolute1e-12 for floating-point batch effects.
Threshold equality accepts; review-all and inactive-policy states always need review.

One vectorized model operation per bounded batch, serialized by the existing lock,
threadpool limit1. No parallel row fan-out, artifact reads, reload, fit or network
provider calls. Calibrated mode preserves the already-frozen temperature/threshold.

The new batch POST has an **8 MiB** received-byte ceiling before JSON parsing,
covering 100x5000 characters even as escaped astral Unicode (~6 MB), plus framing
headroom. Content-Length is checked early; actual streamed bytes are also counted.
413 returns typed payload_too_large. Compressed bodies return415 unsupported_encoding;
invalid JSON, duplicate keys and NaN/Infinity return sanitized422. Structural JSON
schema errors also return422 with static locations, no values/unknown keys echoed.
Existing endpoints preserve their contracts. An absent/untrusted model returns503;
inference failure returns sanitized500 and disables model readiness until restart.
A missing/invalid/disabled policy keeps legitimate raw predictions and needs_review.

## CSV interface

Streamlit 1.66.1 or newer is required for the tested per-widget upload cap and
public file-upload AppTest API. The project minimum now matches the installed,
verified 1.66.1 runtime; no model dependency or frozen artifact identity changes.

Single Message and Batch CSV share local API discovery and status. Upload exactly
one UTF-8 CSV (standard BOM allowed), <=1 MiB and 1–200 data rows, header exactly
text. No id or other columns in Phase 5. Quoted commas/newlines are supported;
wrong/duplicate headers, malformed quoting, inconsistent fields, bad encoding,
NUL controls, empty/oversize files or too many rows reject the complete upload.
Blank and >5000-character cells become invalid_text row outcomes; all other valid
rows are normalized with the single-message schema. Blank CSV records with no
field are structural errors; use a quoted empty cell to represent an invalid row.

Only **Classify CSV** posts valid rows in sequential chunks <=100 to the local API.
Upload, rerender and download never trigger prediction. The client disables proxies
and redirects, uses a30 s batch timeout, validates typed responses/counts, and stops
on the first failed chunk. Previous successes stay; every remaining valid row gets
an explicit api_timeout/api_unavailable/api_error/invalid_response status. Invalid
input rows retain invalid_text. Every data row has a stable generated 1-based index.

Export is generated in memory and includes only row_index,status,intent,top_score,
score_type (calibrated/uncalibrated),recommendation,model_version,policy_status,
policy_version,reason. No source text, file name or optional untrusted IDs. Formula
prefixes (=,+,-,@ after whitespace/BOM) and control-bearing output metadata are
apostrophe-escaped before csv.writer quoting. No upload is written to disk or
retained as customer history. Current-session result state stores only the export,
output records and a fingerprint; changing the file or model/policy/API state
clears it. Uploaded contents remain transient in Streamlit memory.

Model not ready disables both classification controls and upload. Model-only
operation gives raw scores and human review. A verified policy provides calibrated
recommendations, and the actual audit warning remains prominent: **10.13% upper
error bound exceeded the chosen 10.00% demo budget**. Ready is an integrity state,
not safety certification. English banking single-intent only, no OOD guarantee.

## Local smoke checks

Start a serving mode using [README](../README.md) / [serving.md](serving.md), with
INTENTOPS_API_PORT=18000 for Compose. From PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:18000/health
Invoke-RestMethod http://127.0.0.1:18000/ready
Invoke-RestMethod http://127.0.0.1:18000/model/info
Invoke-RestMethod http://127.0.0.1:18000/policy/info
Invoke-WebRequest http://127.0.0.1:18000/docs
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:18000/predict -ContentType 'application/json' -Body '{"text":"When will my new card arrive?"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:18000/predict/batch -ContentType 'application/json' -Body '{"messages":["When will my new card arrive?","Hello, please help."]}'
```

Open http://127.0.0.1:8501, select Batch CSV, download the fictional sample, upload
it and click Classify CSV. Inspect the table, click Download results CSV and parse
or open it: three row indexes, no source-text column. An added quoted empty cell
should remain an invalid_text row while the other three classify. API/down and
file-size/row-limit errors must be displayed rather than fabricated predictions.
See [benchmarks](benchmarks.md) for real measured HTTP workload instructions.

All public fixtures/examples are invented. Official test data stays reserved;
no model/policy fitting or new audit occurs in Phase 5. MIT code and separate
BANKING77 CC BY 4.0 obligations remain in [third-party notices](../THIRD_PARTY_NOTICES.md).
