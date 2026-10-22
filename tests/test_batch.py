"""Fictional bounded batch contracts, vectorization and privacy failure injection."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_policy import ready_service as ready_service  # noqa: PLC0414 -- shared fixture.
from test_serving import bundle as bundle  # noqa: PLC0414 -- shared fixture.
from test_serving import deny_external as deny_external  # noqa: PLC0414 -- shared fixture.

from app.api import create_app
from app.inference import ModelService
from app.request_limits import MAX_BATCH_BODY_BYTES, BatchBodyLimit
from app.schemas import BatchPredictionResponse

pytestmark = pytest.mark.usefixtures("deny_external")


@pytest.mark.parametrize("policy", [True, False])
def test_batch_single_equivalence_order_and_schema(ready_service, policy):
    service, _ = ready_service
    if not policy:
        service.load_policy(Path("not-read"), enabled=False)
    messages = ["fictional orchard apple", "fictional galaxy rocket", "fictional orchard apple"]
    with TestClient(create_app(lambda: service)) as client:
        batch = client.post("/predict/batch", json={"messages": messages})
        assert batch.status_code == 200
        result = BatchPredictionResponse.model_validate(batch.json())
        for message, actual in zip(messages, result.predictions, strict=True):
            single = client.post("/predict", json={"text": message}).json()
            assert [p.score for p in actual.top_predictions] == pytest.approx(
                [p["score"] for p in single.pop("top_predictions")], abs=1e-12
            )
            assert actual.model_dump(exclude={"top_predictions"}) == single
        if not policy:
            assert all(
                r.recommendation == "needs_review"
                and r.score_type == "uncalibrated_model_probability"
                for r in result.predictions
            )
        schema = client.get("/openapi.json").json()
        array = schema["components"]["schemas"]["BatchPredictionRequest"]["properties"]["messages"]
        assert array["minItems"] == 1 and array["maxItems"] == 100
        assert array["items"]["maxLength"] == 5000
        assert "413" in schema["paths"]["/predict/batch"]["post"]["responses"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [],
        {"messages": []},
        {"messages": ["a"] * 101},
        {"messages": "private-message"},
        {"messages": [None]},
        {"messages": [True]},
        {"messages": [42]},
        {"messages": ["a", " "]},
        {"messages": ["x" * 5001]},
        {"messages": ["a"], "private-key": "secret"},
    ],
)
def test_invalid_batch_all_or_nothing_and_sanitized(payload, monkeypatch):
    service = ModelService()

    def forbidden(*a):
        raise AssertionError("Invalid batch must not reach inference")

    monkeypatch.setattr(service, "predict_batch", forbidden)
    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422
        assert "private-key" not in response.text and "private-message" not in response.text
        assert "secret" not in response.text and "x" * 50 not in response.text
        assert all(set(issue) == {"loc", "msg", "type"} for issue in response.json()["detail"])


@pytest.mark.parametrize(
    "body",
    [
        b'{"messages":["a"]',
        b'{"messages":[NaN]}',
        b'{"messages":[Infinity]}',
        b'{"messages":["a"],"messages":["b"]}',
        b"\xff",
        b"[" * 1100 + b"]" * 1100,
    ],
    ids=lambda body: f"json-{len(body)}-bytes",
)
def test_json_edge_cases_sanitized(body):
    with TestClient(create_app(lambda: ModelService())) as client:
        response = client.post(
            "/predict/batch", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422 and response.json()["detail"]
        assert len(response.content) < 1000 and "NaN" not in response.text


def test_no_model_and_invalid_policy_fallback(ready_service):
    with TestClient(create_app(lambda: ModelService())) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict/batch", json={"messages": ["fictional"]}).status_code == 503
    service, root = ready_service
    (root / "current.json").write_text("{}")
    service.load_policy(root, enabled=True)
    with TestClient(create_app(lambda: service)) as client:
        result = client.post("/predict/batch", json={"messages": ["fictional", "fictional two"]})
        assert result.status_code == 200 and client.get("/ready").status_code == 200
        assert all(
            r["policy_status"] == "invalid" and r["recommendation"] == "needs_review"
            for r in result.json()["predictions"]
        )


def test_vectorized_once_no_runtime_reads_or_refits(bundle, monkeypatch):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)
    original = service.model.predict_proba
    calls = []

    def counted(texts):
        calls.append(len(texts))
        return original(texts)

    monkeypatch.setattr(service.model, "predict_proba", counted)
    with TestClient(create_app(lambda: service)) as client:

        def forbidden(*a, **k):
            raise AssertionError("Batch must not open files, refit or unpickle")

        monkeypatch.setattr(Path, "open", forbidden)
        monkeypatch.setattr(service.model, "fit", forbidden)
        response = client.post("/predict/batch", json={"messages": ["fictional orchard"] * 100})
        assert response.status_code == 200 and len(response.json()["predictions"]) == 100
    assert calls == [100]


def test_ties_threshold_equality_and_review_all(bundle, monkeypatch):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)
    monkeypatch.setattr(
        service.model,
        "predict_proba",
        lambda texts: np.tile([0.5, 0.25, 0.125, 0.125], (len(texts), 1)),
    )
    monkeypatch.setattr(
        service.model,
        "decision_function",
        lambda texts: np.log(np.tile([0.5, 0.25, 0.125, 0.125], (len(texts), 1))),
    )
    service.policy_version = "fictional-policy"
    service.policy_status = "ready"
    for threshold in (0.5, None):
        service.policy = SimpleNamespace(
            frozen=SimpleNamespace(temperature=1.0, selected=SimpleNamespace(threshold=threshold)),
            audit_risk=SimpleNamespace(rows=100, accepted=50, wilson_upper=0.1),
        )
        result = service.predict_batch(["fictional"] * 2)
        assert all(r.intent == data.labels[0] for r in result.predictions)
        assert [p.intent for p in result.predictions[0].top_predictions] == data.labels[:3]
        assert all(
            r.recommendation == ("suggest_automatic" if threshold else "needs_review")
            for r in result.predictions
        )


@pytest.mark.parametrize("fault", ["exception", "nan", "wrong_rows"])
def test_batch_model_failure_closes_readiness_without_text(bundle, monkeypatch, fault, caplog):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)

    def failed(texts):
        if fault == "exception":
            raise RuntimeError("private-input /private/path")
        return np.full((len(texts) if fault == "nan" else 1, 4), np.nan if fault == "nan" else 0.25)

    monkeypatch.setattr(service.model, "predict_proba", failed)
    with TestClient(create_app(lambda: service)) as client:
        response = client.post("/predict/batch", json={"messages": ["private-input", "fictional"]})
        assert response.status_code == 500 and "private-input" not in response.text
        assert client.get("/ready").status_code == 503
    assert "private-input" not in caplog.text and "/private/path" not in caplog.text


def test_body_limit_and_uncompressed_only():
    with TestClient(create_app(lambda: ModelService())) as client:
        response = client.post(
            "/predict/batch",
            content=b"x",
            headers={"Content-Length": str(MAX_BATCH_BODY_BYTES + 1)},
        )
        assert (
            response.status_code == 413 and response.json()["error"]["code"] == "payload_too_large"
        )
        assert (
            client.post(
                "/predict/batch", content=b"x", headers={"Content-Encoding": "gzip"}
            ).status_code
            == 415
        )
        # Existing single endpoint retains its original contract.
        assert client.post("/predict", json={"text": "fictional"}).status_code == 503


def test_streamed_limit_counts_actual_bytes_and_stops_reading(monkeypatch):
    monkeypatch.setattr("app.request_limits.MAX_BATCH_BODY_BYTES", 20)

    async def run():
        sent = []
        received = []
        events = iter(
            [
                {"type": "http.request", "body": b"x" * 15, "more_body": True},
                {"type": "http.request", "body": b"x" * 15, "more_body": True},
            ]
        )

        async def receive():
            received.append(1)
            return next(events)

        async def send(event):
            sent.append(event)

        async def forbidden(*a):
            raise AssertionError("Oversize body reached parser/inference")

        await BatchBodyLimit(forbidden)(
            {
                "type": "http",
                "method": "POST",
                "path": "/predict/batch",
                "headers": [(b"content-length", b"0")],
            },
            receive,
            send,
        )
        assert sent[0]["status"] == 413 and len(received) == 2

    asyncio.run(run())


def test_maximum_unicode_json_fits_documented_ceiling(bundle, monkeypatch):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)
    monkeypatch.setattr(
        service.model, "predict_proba", lambda texts: np.full((len(texts), 4), 0.25)
    )
    body = json.dumps({"messages": ["🚀" * 5000] * 100}).encode()
    assert len(body) < MAX_BATCH_BODY_BYTES
    with TestClient(create_app(lambda: service)) as client:
        response = client.post(
            "/predict/batch", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200 and len(response.json()["predictions"]) == 100
