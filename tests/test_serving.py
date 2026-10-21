"""Serving uses trusted fictional bundles; no BANKING77 or model endpoint access."""

import json
import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.inference import ModelService
from app.schemas import PredictionResponse
from ml import dataset as ds
from ml import model_support as support
from ml import train


@pytest.fixture
def bundle(model_data, tmp_path, monkeypatch):
    data, *_ = model_data
    monkeypatch.setattr(
        train, "save_confusion", lambda metrics, path: path.write_bytes(b"fictional plot")
    )
    model, baseline, timing = train.train_models(data)
    root = tmp_path / "artifacts"
    _, folder = train.publish_run(root, data, model, baseline, timing)
    return root, folder, data


@pytest.fixture
def deny_external(monkeypatch):
    original = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple):
            assert address[0] in ("127.0.0.1", "::1", "localhost"), "External connection forbidden"
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)


def test_ready_prediction_matches_full_pipeline(bundle, deny_external):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").json() == {"status": "ok", "model_ready": True}
        assert client.get("/ready").status_code == 200
        info = client.get("/model/info").json()
        assert info["status"] == "loaded" and info["category_count"] == 4
        text = "fictional orchard apple request"
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        result = PredictionResponse.model_validate(response.json())
        values = service.model.predict_proba([text])[0]
        indices = np.argsort(-values, kind="stable")[:3]
        assert [p.intent for p in result.top_predictions] == [
            str(service.model.classes_[i]) for i in indices
        ]
        assert [p.score for p in result.top_predictions] == pytest.approx(values[indices])
        assert result.intent == service.model.predict([text])[0]
        assert text not in response.text
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/predict"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/PredictionResponse")


def test_untrusted_default_no_deserialization(monkeypatch):
    monkeypatch.delenv("INTENTOPS_TRUST_LOCAL_ARTIFACTS", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Untrusted loader called")

    monkeypatch.setattr("app.inference.load_artifacts", forbidden)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"text": "fictional request"}).status_code == 503
        assert client.get("/model/info").json()["reason"] == "disabled"


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_only_explicit_one_enables_trust(monkeypatch, value):
    monkeypatch.setenv("INTENTOPS_TRUST_LOCAL_ARTIFACTS", value)

    def forbidden(*args, **kwargs):
        raise AssertionError("Loader should remain disabled")

    monkeypatch.setattr("app.inference.load_artifacts", forbidden)
    assert not ModelService.from_environment().ready


def test_load_once_no_runtime_file_reads_or_hot_swap(bundle, monkeypatch):
    root, _, data = bundle
    calls = []

    def factory():
        calls.append(1)
        return ModelService.load(root, trusted=True, contract=data)

    with TestClient(create_app(factory)) as client:
        expected = client.get("/model/info").json()
        (root / "current.json").write_text("{}")

        def forbidden(*args, **kwargs):
            raise AssertionError("Dataset/artifact read during request")

        monkeypatch.setattr(ds, "read_bytes", forbidden)
        for _ in range(3):
            assert client.get("/model/info").json() == expected
            assert client.post("/predict", json={"text": "fictional orchard"}).status_code == 200
    assert calls == [1]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_pointer",
        "missing_model",
        "corrupt",
        "unexpected",
        "path",
        "python",
        "package",
        "config",
        "inventory",
        "size",
    ],
)
def test_bad_bundles_fail_before_deserialization(bundle, monkeypatch, failure, caplog):
    root, folder, data = bundle
    pointer = root / "current.json"
    if failure == "missing_pointer":
        pointer.unlink()
    elif failure == "missing_model":
        (folder / "intent_classifier.joblib").unlink()
    elif failure == "corrupt":
        (folder / "intent_classifier.joblib").write_bytes(b"fictional corruption")
    elif failure == "unexpected":
        (folder / "extra.txt").write_text("fictional extra")
    elif failure == "path":
        pointer.write_bytes(ds.json_bytes({"run": "../private-path"}))
    elif failure == "size":
        monkeypatch.setattr(support, "MAX_ARTIFACT_BYTES", 2)
    else:
        path = folder / "model_metadata.json"
        meta = json.loads(path.read_text())
        if failure == "python":
            meta["versions"]["python"] = "0.0.0"
        elif failure == "package":
            meta["versions"]["sklearn"] = "0.0.0"
        elif failure == "config":
            meta["configuration"]["classifier"]["C"] = 999
        elif failure == "inventory":
            meta["artifact_sha256"]["extra.joblib"] = "0" * 64
        path.write_bytes(ds.json_bytes(meta))
        p = json.loads(pointer.read_text())
        p["metadata_sha256"] = ds.sha256(path.read_bytes())
        pointer.write_bytes(ds.json_bytes(p))

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected bundle must not deserialize")

    monkeypatch.setattr(support.joblib, "load", forbidden)
    # Assert the verifier really rejects; the outer service also sanitizes failures.
    with pytest.raises((ds.DataError, OSError)):
        support.load_artifacts(root, data, trusted=True)
    service = ModelService.load(root, trusted=True, contract=data)
    assert not service.ready
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        response = client.post("/predict", json={"text": "fictional private request"})
        assert response.status_code == 503
        assert str(root) not in response.text and "fictional private request" not in response.text
    assert str(root) not in caplog.text


def test_tie_order_is_declared_class_order(bundle, monkeypatch):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)
    monkeypatch.setattr(service.model, "predict_proba", lambda texts: np.full((1, 4), 0.25))
    assert [p.intent for p in service.predict("fictional").top_predictions] == data.labels[:3]


@pytest.mark.parametrize("fault", ["nan", "negative", "exception"])
def test_unexpected_model_failure_sanitized(bundle, monkeypatch, fault, caplog):
    root, _, data = bundle
    service = ModelService.load(root, trusted=True, contract=data)

    def failed(texts):
        if fault == "exception":
            raise RuntimeError("fictional private message /secret/path")
        return (
            np.array([[float("nan"), 0, 0, 1]])
            if fault == "nan"
            else np.array([[-0.1, 0.3, 0.4, 0.4]])
        )

    monkeypatch.setattr(service.model, "predict_proba", failed)
    with TestClient(create_app(lambda: service)) as client:
        result = client.post("/predict", json={"text": "fictional private message"})
        assert result.status_code == 500 and result.json()["error"]["code"] == "inference_failed"
        assert "fictional" not in result.text and "/secret/path" not in result.text
        assert client.get("/ready").status_code == 503
        assert client.get("/health").json()["model_ready"] is False
    assert "fictional private message" not in caplog.text and "/secret/path" not in caplog.text


def test_no_private_extra_field_name_echoed():
    with TestClient(create_app(lambda: ModelService())) as client:
        response = client.post(
            "/predict", json={"text": "fictional", "private-customer-string": "value"}
        )
        assert response.status_code == 422 and "private-customer-string" not in response.text


def test_serving_contract_requires_no_dataset_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Serving must not read datasets")

    monkeypatch.setattr(ds, "read_bytes", forbidden)
    contract = support.serving_contract()
    assert contract.train_text == [] and len(contract.labels) == 77
    assert contract.provenance["reserved_test_content_read"] is False
