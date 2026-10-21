"""HTTP contracts without a server, network, dataset, or model."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import app
from app.schemas import HealthResponse, ModelInfoResponse, PredictionRequest

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": False}


def test_model_info() -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json() == {
        "phase": 0,
        "status": "untrained",
        "model_ready": False,
        "model_version": None,
    }


@pytest.mark.parametrize("text", ["Where is my card?", "x" * 5000])
def test_prediction_unavailable(text: str) -> None:
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "model_not_ready",
            "message": "No trained model is available. Classification is deferred to a later phase.",
        }
    }
    assert text not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": " \n\t"},
        {"text": "x" * 5001},
        {"text": 123},
        {"text": None},
        {"text": True},
        {"text": ["hello"]},
        {"text": "hello", "extra": 1},
    ],
)
def test_invalid_prediction(payload: dict) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert all(set(detail) == {"loc", "type", "msg"} for detail in response.json()["detail"])


def test_request_normalizes_whitespace() -> None:
    assert PredictionRequest(text="  example  ").text == "example"


@pytest.mark.parametrize(
    "schema, payload",
    [
        (HealthResponse, {"model_ready": True}),
        (HealthResponse, {"extra": "value"}),
        (ModelInfoResponse, {"status": "trained"}),
        (ModelInfoResponse, {"model_version": "fake"}),
    ],
)
def test_response_schemas_reject_false_claims(schema: type, payload: dict) -> None:
    with pytest.raises(ValidationError):
        schema(**payload)


def test_validation_does_not_echo_private_message() -> None:
    response = client.post(
        "/predict", json={"text": "private-example", "secret": "private-example"}
    )
    assert response.status_code == 422
    assert "private-example" not in response.text
