"""Dashboard states and real typed prediction rendering using local fake HTTP."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

SCRIPT = Path(__file__).resolve().parents[1] / "dashboard/streamlit_app.py"


def setup_http(monkeypatch, *, ready=True, failure=None, policy_mode="disabled", automatic=True):
    def get(client, path):
        if failure == "down":
            raise httpx.ConnectError("fictional internal path must not display")
        if failure == "status_timeout":
            raise httpx.ReadTimeout("fictional timeout")
        if path == "/policy/info":
            policy_body = {"status": "disabled"}
            if policy_mode == "ready":
                policy_body = {
                    "status": "ready",
                    "policy_version": "review-fixture",
                    "method": "temperature_scaling_v1",
                    "temperature": 0.5,
                    "threshold": 0.6,
                    "mode": "threshold",
                    "demo_error_budget": 0.1,
                    "audit_rows": 100,
                    "audit_accepted": 80,
                    "audit_upper_error_bound": 0.11,
                }
            elif policy_mode == "malformed":
                policy_body = {"status": "ready", "temperature": "private detail"}
            return httpx.Response(
                200, json=policy_body, request=httpx.Request("GET", "http://api" + path)
            )
        body = (
            {"status": "ok", "model_ready": ready}
            if path == "/health"
            else {
                "phase": 3,
                "status": "loaded" if ready else "unavailable",
                "model_ready": ready,
                "model_version": "fixture-v1" if ready else None,
                "classifier_type": "TfidfVectorizer + LogisticRegression" if ready else None,
                "category_count": 4 if ready else 0,
                "reason": None if ready else "disabled",
            }
        )
        return httpx.Response(200, json=body, request=httpx.Request("GET", "http://api" + path))

    calls = []

    def post(client, path, *, json):
        calls.append(json)
        if failure == "timeout":
            raise httpx.ReadTimeout("fictional private detail")
        if failure in ("503", "422", "500"):
            return httpx.Response(
                int(failure),
                json={"error": "must not echo"},
                request=httpx.Request("POST", "http://api/predict"),
            )
        body = {
            "intent": "alpha_case",
            "model_version": "fixture-v1",
            "score_type": "uncalibrated_model_probability",
            "top_predictions": [
                {"intent": "alpha_case", "score": 0.7},
                {"intent": "Beta?", "score": 0.2},
                {"intent": "gamma_case", "score": 0.1},
            ],
        }
        if policy_mode == "ready":
            body.update(
                {
                    "score_type": "temperature_scaled_probability",
                    "policy_status": "ready",
                    "policy_version": "review-fixture",
                    "policy_threshold": 0.6 if automatic else 0.9,
                    "recommendation": "suggest_automatic" if automatic else "needs_review",
                    "reason": "meets_configured_threshold"
                    if automatic
                    else "below_configured_threshold",
                }
            )
        if failure == "malformed":
            body = {"intent": "fake"}
        return httpx.Response(200, json=body, request=httpx.Request("POST", "http://api/predict"))

    monkeypatch.setattr(httpx.Client, "get", get)
    monkeypatch.setattr(httpx.Client, "post", post)
    return calls


@pytest.mark.parametrize("available", [True, False])
def test_dashboard_status(monkeypatch, available):
    setup_http(monkeypatch, ready=False, failure=None if available else "down")
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    assert not view.exception and view.button[0].disabled
    assert not view.dataframe
    if available:
        assert view.warning[0].value == "API online — Model unavailable"
    else:
        assert "API is unavailable" in view.warning[0].value


def test_ready_dashboard_classifies_on_explicit_click(monkeypatch):
    calls = setup_http(monkeypatch)
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    assert view.button[0].disabled
    view.text_area[0].set_value("  fictional orchard request  ").run()
    assert not view.button[0].disabled and calls == []
    view.button[0].click().run()
    assert not view.exception
    assert calls == [{"text": "fictional orchard request"}]
    assert view.success[-1].value == "alpha_case"
    assert len(view.dataframe) == 1
    assert "Uncalibrated probability" in view.dataframe[0].value.columns


@pytest.mark.parametrize(
    "failure,fragment",
    [
        ("503", "Model unavailable"),
        ("422", "Invalid input"),
        ("500", "inference error"),
        ("timeout", "timed out"),
        ("malformed", "invalid prediction response"),
    ],
)
def test_prediction_errors_sanitized(monkeypatch, failure, fragment):
    setup_http(monkeypatch, failure=failure)
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    view.text_area[0].set_value("fictional request").run()
    view.button[0].click().run()
    assert not view.exception and not view.dataframe
    messages = [e.value for e in view.error] + [w.value for w in view.warning]
    assert any(fragment in m for m in messages)
    assert all("fictional private detail" not in m for m in messages)


def test_external_api_configuration_rejected(monkeypatch):
    monkeypatch.setenv("INTENTOPS_API_URL", "https://example.com")

    def blocked(*args, **kwargs):
        raise AssertionError("External network forbidden")

    monkeypatch.setattr(httpx.Client, "get", blocked)
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    assert view.button[0].disabled and view.error


def test_status_timeout_disables_classification(monkeypatch):
    setup_http(monkeypatch, failure="status_timeout")
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    assert view.button[0].disabled and "timed out" in view.warning[0].value


def test_malformed_status_is_not_model_readiness(monkeypatch):
    def bad(client, path):
        return httpx.Response(
            200,
            json={"private": "fictional detail"},
            request=httpx.Request("GET", "http://api" + path),
        )

    monkeypatch.setattr(httpx.Client, "get", bad)
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    assert view.button[0].disabled and "status response" in view.error[0].value
    assert "fictional detail" not in view.error[0].value


def test_blank_input_disabled_even_with_ready_model(monkeypatch):
    setup_http(monkeypatch)
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    view.text_area[0].set_value("   ").run()
    assert view.button[0].disabled


@pytest.mark.parametrize("automatic", [True, False])
def test_policy_ready_recommendation_and_calibrated_scores(monkeypatch, automatic):
    calls = setup_http(monkeypatch, policy_mode="ready", automatic=automatic)
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    view.text_area[0].set_value("fictional message").run()
    view.button[0].click().run()
    assert not view.exception and calls == [{"text": "fictional message"}]
    assert "Temperature-scaled probability" in view.dataframe[0].value.columns
    messages = [x.value for x in view.info] + [x.value for x in view.warning]
    assert any(
        ("Suggested automatic" if automatic else "below calibrated threshold") in x
        for x in messages
    )


def test_malformed_policy_status_preserves_raw_model_prediction(monkeypatch):
    setup_http(monkeypatch, policy_mode="malformed")
    view = AppTest.from_file(SCRIPT).run(timeout=20)
    assert any("Policy status unavailable or invalid" in x.value for x in view.warning)
    view.text_area[0].set_value("fictional message").run()
    view.button[0].click().run()
    assert not view.exception and view.dataframe
    assert any("no validated review policy" in x.value for x in view.warning)
    assert all("private detail" not in x.value for x in view.warning)
