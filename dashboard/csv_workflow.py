"""Bounded in-memory CSV parsing, chunked local inference and text-free export."""

import csv
import io
from dataclasses import asdict, dataclass
from typing import Literal

import httpx
from pydantic import ValidationError

from app.schemas import MAX_BATCH_MESSAGES, BatchPredictionResponse, PredictionRequest

MAX_CSV_BYTES = 1024 * 1024
MAX_CSV_ROWS = 200
FICTIONAL_CSV = (
    b"text\n"
    b"When will my new card arrive?\n"
    b"I was charged twice for the same card payment.\n"
    b'"Hello, please help."\n'
)


class CSVError(ValueError):
    """Safe fixed operator message; never include file names or input values."""


@dataclass(frozen=True)
class CSVRow:
    row_index: int
    text: str | None


Status = Literal[
    "classified", "invalid_text", "api_timeout", "api_unavailable", "api_error", "invalid_response"
]


@dataclass(frozen=True)
class CSVResult:
    row_index: int
    status: Status
    intent: str | None = None
    top_score: float | None = None
    score_type: str | None = None
    recommendation: str | None = None
    model_version: str | None = None
    policy_status: str | None = None
    policy_version: str | None = None
    reason: str | None = None


def parse_csv(content: bytes) -> list[CSVRow]:
    if not content or len(content) > MAX_CSV_BYTES:
        raise CSVError("Upload a nonempty CSV no larger than 1 MiB.")
    try:
        text = content.decode("utf-8-sig")
        if "\x00" in text:
            raise CSVError("CSV contains unsupported control characters.")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        if next(reader, None) != ["text"]:
            raise CSVError("CSV header must be exactly text; no IDs, duplicate or extra columns.")
        rows = []
        for index, values in enumerate(reader, start=1):
            if index > MAX_CSV_ROWS:
                raise CSVError("CSV supports at most 200 data rows.")
            if len(values) != 1:
                raise CSVError("Each CSV data row must have exactly one field.")
            try:
                normalized = PredictionRequest(text=values[0]).text
            except ValidationError:
                normalized = None
            rows.append(CSVRow(index, normalized))
        if not rows:
            raise CSVError("CSV must contain at least one data row.")
        return rows
    except UnicodeError:
        raise CSVError("CSV must use UTF-8 encoding (BOM is allowed).") from None
    except csv.Error:
        raise CSVError("Malformed CSV quoting or excessive field length.") from None


def classify_rows(rows: list[CSVRow], client: httpx.Client) -> list[CSVResult]:
    if not 1 <= len(rows) <= MAX_CSV_ROWS:
        raise CSVError("CSV supports 1–200 data rows.")
    valid = [r for r in rows if r.text is not None]
    by_index = {r.row_index: CSVResult(r.row_index, "invalid_text") for r in rows}
    for start in range(0, len(valid), MAX_BATCH_MESSAGES):
        chunk = valid[start : start + MAX_BATCH_MESSAGES]
        failure = None
        try:
            response = client.post("/predict/batch", json={"messages": [r.text for r in chunk]})
            response.raise_for_status()
            result = BatchPredictionResponse.model_validate(response.json())
            if len(result.predictions) != len(chunk):
                raise ValueError("Wrong result count")
        except httpx.TimeoutException:
            failure = "api_timeout"
        except httpx.HTTPStatusError as exc:
            failure = "api_unavailable" if exc.response.status_code == 503 else "api_error"
        except httpx.HTTPError:
            failure = "api_unavailable"
        except ValueError:
            failure = "invalid_response"
        if failure:
            # Keep earlier successes, stop further network calls and mark every remaining row.
            for row in valid[start:]:
                by_index[row.row_index] = CSVResult(row.row_index, failure)
            break
        for row, prediction in zip(chunk, result.predictions, strict=True):
            by_index[row.row_index] = CSVResult(
                row.row_index,
                "classified",
                prediction.intent,
                prediction.top_predictions[0].score,
                "calibrated"
                if prediction.score_type == "temperature_scaled_probability"
                else "uncalibrated",
                prediction.recommendation,
                prediction.model_version,
                prediction.policy_status,
                prediction.policy_version,
                prediction.reason,
            )
    return [by_index[r.row_index] for r in rows]


def result_records(results: list[CSVResult]) -> list[dict]:
    return [asdict(r) for r in results]


def _safe_cell(value):
    if isinstance(value, str) and (
        value.lstrip(" \t\r\n\ufeff").startswith(("=", "+", "-", "@"))
        or any(ord(c) < 32 for c in value)
    ):
        return "'" + value
    return value


def export_csv(results: list[CSVResult]) -> bytes:
    """Generated row numbers and prediction fields only. No source texts or IDs."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(CSVResult.__dataclass_fields__), lineterminator="\n"
    )
    writer.writeheader()
    for record in result_records(results):
        writer.writerow({key: _safe_cell(value) for key, value in record.items()})
    return stream.getvalue().encode("utf-8")
