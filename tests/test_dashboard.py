"""Execute dashboard rendering with local HTTP calls replaced by fixtures."""

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("available", [True, False])
def test_dashboard_status(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    def get(client: httpx.Client, path: str) -> httpx.Response:
        if not available:
            raise httpx.ConnectError("Local API unavailable")
        body = (
            {"status": "ok", "model_ready": False}
            if path == "/health"
            else {"phase": 0, "status": "untrained", "model_ready": False, "model_version": None}
        )
        return httpx.Response(200, json=body, request=httpx.Request("GET", "http://api" + path))

    monkeypatch.setattr(httpx.Client, "get", get)
    script = Path(__file__).resolve().parents[1] / "dashboard" / "streamlit_app.py"
    view = AppTest.from_file(script).run(timeout=20)
    assert not view.exception
    assert view.warning[0].value == "Phase 0 — Model not trained"
    assert view.button[0].disabled
    if available:
        assert len(view.json) == 1
    else:
        assert "local API is unavailable" in view.info[0].value
