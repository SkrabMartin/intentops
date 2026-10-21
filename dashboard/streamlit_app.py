"""Local API dashboard; customer text is submitted only on an explicit Classify click."""

import ipaddress
import os

import httpx
import streamlit as st
from pydantic import ValidationError

from app.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse

st.set_page_config(page_title="IntentOps", page_icon="💬")
st.title("IntentOps")
st.caption("Open-source, local-first customer support intent classification")
base_url = os.environ.get("INTENTOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")
ready = False
info = None
try:
    url = httpx.URL(base_url)
    try:
        local = ipaddress.ip_address(url.host).is_loopback
    except ValueError:
        local = url.host in ("localhost", "api")
    if url.scheme != "http" or not local or url.userinfo or url.query or url.path not in ("", "/"):
        raise ValueError("Only local HTTP API addresses are allowed")
    with httpx.Client(
        base_url=base_url, timeout=5.0, trust_env=False, follow_redirects=False
    ) as client:
        response = client.get("/health")
        response.raise_for_status()
        health = HealthResponse.model_validate(response.json())
        response = client.get("/model/info")
        response.raise_for_status()
        info = ModelInfoResponse.model_validate(response.json())
    if health.model_ready != info.model_ready:
        raise ValueError("Inconsistent readiness")
    ready = info.model_ready
    if ready:
        st.success("API online — Model ready")
        st.caption(f"Model: {info.model_version} · {info.category_count} categories")
    else:
        st.warning("API online — Model unavailable")
        st.info(
            "Enable trusted, compatible local artifacts and restart the API. No prediction is generated while unavailable."
        )
except httpx.TimeoutException:
    st.warning("Local API timed out. Check the service and try again.")
except httpx.HTTPError:
    st.warning("The local API is unavailable. Start the API, then refresh this page.")
except (ValueError, ValidationError):
    st.error(
        "Invalid local API configuration or status response. Use a loopback HTTP URL or http://api:8000 in Compose."
    )

text = st.text_area("Customer message", max_chars=5000, placeholder="Use a fictional example.")
try:
    payload = PredictionRequest(text=text)
    valid = True
except ValidationError:
    payload = None
    valid = False
st.caption(
    "Enter 1–5000 nonblank characters. Text is sent only to the local API when you click Classify; it is not saved."
)
if st.button("Classify", disabled=not ready or not valid):
    try:
        with httpx.Client(
            base_url=base_url, timeout=5.0, trust_env=False, follow_redirects=False
        ) as client:
            response = client.post("/predict", json=payload.model_dump())
            response.raise_for_status()
            prediction = PredictionResponse.model_validate(response.json())
        st.subheader("Predicted intent")
        st.success(prediction.intent)
        st.caption(f"Model: {prediction.model_version}")
        st.dataframe(
            [
                {"Intent": p.intent, "Uncalibrated probability": p.score}
                for p in prediction.top_predictions
            ],
            hide_index=True,
        )
    except httpx.TimeoutException:
        st.error("Classification timed out. Try again after checking the local API.")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            st.warning("Model unavailable. Check local artifacts and restart the API.")
        elif exc.response.status_code == 422:
            st.error("Invalid input. Enter 1–5000 nonblank characters.")
        else:
            st.error(
                "The local API reported an inference error. Inspect the service and restart if needed."
            )
    except httpx.HTTPError:
        st.error("The local API is unreachable. Check the service before retrying.")
    except (ValueError, ValidationError):
        st.error("The local API returned an invalid prediction response. No result is displayed.")
st.caption(
    "Scores are uncalibrated model probabilities, not correctness guarantees. No automatic routing or review decision is made."
)
