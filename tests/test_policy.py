"""Fictional policy integrity, fail-closed API integration, and immutable startup."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_calibration import experiment as experiment  # noqa: PLC0414 -- shared pytest fixture.
from test_serving import bundle as bundle  # noqa: PLC0414 -- shared pytest fixture.
from test_serving import deny_external as deny_external  # noqa: PLC0414 -- shared pytest fixture.

from app.api import create_app
from app.inference import ModelService
from app.schemas import PredictionResponse
from ml import calibrate
from ml import calibration as cal
from ml import dataset as ds
from ml import review_policy as store

pytestmark = pytest.mark.usefixtures("deny_external")


@pytest.fixture
def completed(experiment):
    model, metadata, rows, _ = experiment
    frozen = calibrate.develop(model, metadata, rows)
    return calibrate.audit(model, metadata, rows, frozen), metadata


@pytest.fixture
def policy_root(completed, tmp_path):
    record, metadata = completed
    root = tmp_path / "policies"
    version = store.publish(root, record)
    return root, record, metadata, version


def test_json_policy_roundtrip_no_executable_serializer(policy_root):
    root, record, metadata, version = policy_root
    actual, actual_version = store.load_policy(root, metadata)
    assert actual == record and actual_version == version
    assert version.startswith("review-")
    assert not list(root.rglob("*.joblib"))


@pytest.mark.parametrize(
    "failure",
    [
        "digest",
        "pointer",
        "missing",
        "extra",
        "nonfile",
        "size",
        "nan",
        "infinity",
        "negative_threshold",
        "temperature",
        "model",
        "versions",
        "class_order",
        "data",
        "schema",
        "method",
        "membership",
        "selection_rule",
        "audit",
        "duplicate",
        "malformed",
    ],
)
def test_invalid_policy_rejected(policy_root, monkeypatch, failure):
    root, _, metadata, _ = policy_root
    pointer, _ = store.read_json(root / "current.json")
    folder = root / pointer["run"]
    path = folder / "record.json"
    record = json.loads(path.read_bytes())
    if failure == "digest":
        path.write_bytes(b"{}")
    elif failure == "pointer":
        pointer["run"] = "../elsewhere"
    elif failure == "missing":
        path.unlink()
    elif failure == "extra":
        (folder / "extra.json").write_text("{}")
    elif failure == "nonfile":
        (folder / "unexpected").mkdir()
    elif failure == "size":
        monkeypatch.setattr(store, "MAX_POLICY_BYTES", 300)
    else:
        if failure == "nan":
            record["frozen"]["temperature"] = float("nan")
        elif failure == "infinity":
            record["frozen"]["temperature"] = float("inf")
        elif failure == "negative_threshold":
            record["frozen"]["selected"]["threshold"] = -0.1
        elif failure == "temperature":
            record["frozen"]["temperature"] = 0.0
        elif failure in ("model", "versions", "class_order", "data"):
            key = {
                "model": "model_version",
                "versions": "versions",
                "class_order": "class_order_sha256",
                "data": "provenance",
            }[failure]
            record["frozen"]["binding"][key] = (
                {"wrong": "value"} if failure in ("versions", "data") else "0" * 64
            )
        elif failure == "schema":
            record["format_version"] = 2
        elif failure == "method":
            record["frozen"]["protocol"]["method"] = "unknown"
        elif failure == "membership":
            record["frozen"]["manifest"]["selection"]["source_ids"][0] = record["frozen"][
                "manifest"
            ]["audit"]["source_ids"][0]
        elif failure == "selection_rule":
            record["frozen"]["selected"]["accepted"] = 99
        elif failure == "audit":
            record["audit_risk"]["rows"] += 1
        if failure not in ("nan", "infinity"):
            record["frozen_sha256"] = ds.sha256(ds.json_bytes(record["frozen"]))
        content = json.dumps(record).encode()
        if failure == "duplicate":
            content = content.replace(
                b'"format_version": 1', b'"format_version": 1, "format_version": 1', 1
            )
        if failure == "malformed":
            content = b"{malformed fictional JSON"
        path.write_bytes(content)
        pointer["sha256"] = ds.sha256(content)
    if failure not in ("digest", "missing", "extra", "nonfile", "size"):
        (root / "current.json").write_bytes(ds.json_bytes(pointer))
    with pytest.raises((ValueError, TypeError, OSError)):
        store.load_policy(root, metadata)


def test_symlinks_rejected(policy_root, tmp_path, monkeypatch):
    root, _, metadata, _ = policy_root
    pointer = root / "current.json"
    target = tmp_path / "pointer.json"
    pointer.rename(target)
    try:
        pointer.symlink_to(target)
    except OSError:
        pointer.write_bytes(target.read_bytes())
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda p: p == pointer or original(p))
    with pytest.raises(ValueError, match="regular policy file"):
        store.load_policy(root, metadata)


def test_atomic_publication_failure_preserves_current(policy_root, monkeypatch):
    root, record, metadata, version = policy_root
    before = (root / "current.json").read_bytes()

    def failed(*a):
        raise OSError("fictional interrupted pointer publication")

    monkeypatch.setattr(ds, "atomic_write", failed)
    with pytest.raises(OSError):
        store.publish(root, record)
    assert (root / "current.json").read_bytes() == before
    assert store.load_policy(root, metadata)[1] == version


@pytest.fixture
def ready_service(bundle, tmp_path):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)
    # A larger fictional validation set for stratified calibration; not official rows.
    rows = [
        {
            "source_row_id": str(i),
            "text": data.train_text[i % len(data.train_text)],
            "category": data.train_labels[i % len(data.train_labels)],
        }
        for i in range(240)
    ]
    metadata = service.metadata.copy()
    # Keep exact original synthetic provenance: use its declared validation size.
    # This fixture needs a complete synthetic model/data binding for 240 validation rows.
    metadata = json.loads(ds.json_bytes(metadata))
    metadata["data_provenance"]["validation_rows"] = 240
    service.metadata = metadata
    frozen = calibrate.develop(service.model, metadata, rows)
    completed = calibrate.audit(service.model, metadata, rows, frozen)
    policy_dir = tmp_path / "active"
    store.publish(policy_dir, completed)
    service.load_policy(policy_dir, enabled=True)
    assert service.policy_status == "ready"
    return service, policy_dir


def test_calibrated_api_matches_pipeline_and_policy(ready_service):
    service, _root = ready_service
    text = "fictional orchard apple"
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/ready").status_code == 200
        info = client.get("/policy/info").json()
        assert info["status"] == "ready"
        result = PredictionResponse.model_validate(
            client.post("/predict", json={"text": text}).json()
        )
        full = cal.scaled(
            service.model.decision_function([text]), service.policy.frozen.temperature
        )[0]
        order = np.argsort(-full, kind="stable")[:3]
        assert [p.score for p in result.top_predictions] == pytest.approx(full[order], abs=1e-12)
        assert result.score_type == "temperature_scaled_probability"
        assert result.intent == service.model.predict([text])[0]
        assert result.recommendation == (
            "suggest_automatic"
            if info["threshold"] is not None and full.max() >= info["threshold"]
            else "needs_review"
        )
        assert "PolicyInfoResponse" in client.get("/openapi.json").json()["components"]["schemas"]


@pytest.mark.parametrize("state", ["disabled", "missing", "invalid"])
def test_model_ready_policy_fallback_raw_and_review(ready_service, tmp_path, state, caplog):
    service, root = ready_service
    if state == "invalid":
        (root / "current.json").write_text('{"path":"private-secret"}')
    service.load_policy(
        tmp_path / "absent" if state == "missing" else root, enabled=state != "disabled"
    )
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/ready").status_code == 200
        result = client.post("/predict", json={"text": "fictional private input"})
        body = result.json()
        assert body["recommendation"] == "needs_review"
        assert body["score_type"] == "uncalibrated_model_probability"
        assert (
            body["policy_status"]
            == {"disabled": "disabled", "missing": "unavailable", "invalid": "invalid"}[state]
        )
        assert "private-secret" not in result.text
    assert "private-secret" not in caplog.text and "fictional private input" not in caplog.text


def test_review_all_active_policy_never_suggests_automatic(ready_service, tmp_path):
    service, _ = ready_service
    rows = [
        {
            "source_row_id": str(i),
            "text": "fictional neutral unknown message",
            "category": list(service.model.classes_)[i % 4],
        }
        for i in range(240)
    ]
    frozen = calibrate.develop(service.model, service.metadata, rows)
    assert frozen.selected.threshold is None
    record = calibrate.audit(service.model, service.metadata, rows, frozen)
    root = tmp_path / "review-all"
    store.publish(root, record)
    service.load_policy(root, enabled=True)
    with TestClient(create_app(lambda: service)) as client:
        info = client.get("/policy/info").json()
        assert info["status"] == "ready" and info["mode"] == "review_all"
        assert info["audit_upper_error_bound"] is None and info["audit_accepted"] == 0
        body = client.post("/predict", json={"text": "fictional orchard"}).json()
        assert body["recommendation"] == "needs_review" and body["reason"] == "review_all_policy"
        assert body["score_type"] == "temperature_scaled_probability"


def test_policy_loaded_once_and_no_request_reads_or_refits(ready_service, monkeypatch):
    service, root = ready_service
    version = service.policy_version
    (root / "current.json").write_text("{}")

    def forbidden(*a, **k):
        raise AssertionError("No request reads or fitting")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(cal, "fit_temperature", forbidden)
    monkeypatch.setattr(service.model, "fit", forbidden)
    for _ in range(3):
        assert service.predict("fictional orchard").policy_version == version


def test_default_disabled_policy_does_not_read(monkeypatch):
    monkeypatch.delenv("INTENTOPS_ENABLE_REVIEW_POLICY", raising=False)
    monkeypatch.delenv("INTENTOPS_TRUST_LOCAL_ARTIFACTS", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("No policy loader without opt-in and model")

    monkeypatch.setattr(store, "load_policy", forbidden)
    service = ModelService.from_environment()
    assert not service.ready
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/policy/info").json()["status"] == "unavailable"


@pytest.mark.parametrize(
    "change",
    [
        {"recommendation": "suggest_automatic"},
        {"score_type": "temperature_scaled_probability"},
        {"policy_status": "ready"},
    ],
)
def test_schema_cannot_claim_automatic_without_policy(change):
    body = {
        "intent": "alpha",
        "model_version": "fictional",
        "top_predictions": [
            {"intent": "alpha", "score": 0.6},
            {"intent": "beta", "score": 0.3},
            {"intent": "gamma", "score": 0.1},
        ],
        **change,
    }
    with pytest.raises(ValueError):
        PredictionResponse.model_validate(body)
