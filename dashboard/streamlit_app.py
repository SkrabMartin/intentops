"""Local API dashboard; customer text is submitted only on an explicit Classify click."""

import os

import httpx
import streamlit as st
from pydantic import ValidationError

from app.local_http import local_api_url
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PolicyInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from dashboard.batch_ui import render_csv

st.set_page_config(page_title="IntentOps", page_icon="💬")
st.title("IntentOps")
st.caption("Open-source, local-first customer support intent classification")
base_url = os.environ.get("INTENTOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")
ready = False
info = None
policy = None
try:
    base_url = local_api_url(base_url)
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

if ready:
    try:
        with httpx.Client(
            base_url=base_url, timeout=5.0, trust_env=False, follow_redirects=False
        ) as client:
            response = client.get("/policy/info")
            response.raise_for_status()
            policy = PolicyInfoResponse.model_validate(response.json())
        st.caption(f"Calibration / review policy: {policy.status}")
        if policy.status == "ready":
            st.caption(f"Policy: {policy.policy_version} · {policy.mode}")
            st.caption(
                f"Internal audit: {policy.audit_accepted}/{policy.audit_rows} accepted; "
                f"one-sided upper error bound: {policy.audit_upper_error_bound:.2%}"
                if policy.audit_upper_error_bound is not None
                else "Internal audit: zero accepted; selective error is undefined."
            )
            if (
                policy.audit_upper_error_bound is not None
                and policy.audit_upper_error_bound > policy.demo_error_budget
            ):
                st.warning(
                    "Internal audit upper error bound exceeds the chosen demonstration budget. This policy does not establish a safety guarantee."
                )
        else:
            st.info(
                "Human review suggested (no validated review policy active). Raw model scores remain available."
            )
    except (httpx.HTTPError, ValueError, ValidationError):
        st.warning(
            "Policy status unavailable or invalid. Predictions must carry a valid recommendation; no action is executed."
        )

single_tab, csv_tab = st.tabs(["Single Message", "Batch CSV"])
with single_tab:
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
            if prediction.policy_status != "ready":
                st.warning("Human review suggested (no validated review policy active)")
            elif prediction.recommendation == "suggest_automatic":
                st.info(
                    "Recommendation: Suggested automatic classification — meets configured development threshold"
                )
            else:
                st.warning(
                    "Recommendation: Human review — "
                    + (
                        "review-all policy"
                        if prediction.reason == "review_all_policy"
                        else "below calibrated threshold"
                    )
                )
            st.caption(
                f"Policy status: {prediction.policy_status} · Policy: {prediction.policy_version or 'none'}"
            )
            score_label = (
                "Temperature-scaled probability"
                if prediction.score_type == "temperature_scaled_probability"
                else "Uncalibrated probability"
            )
            st.dataframe(
                [{"Intent": p.intent, score_label: p.score} for p in prediction.top_predictions],
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
            st.error(
                "The local API returned an invalid prediction response. No result is displayed."
            )
with csv_tab:
    render_csv(
        base_url,
        ready=ready,
        identity={
            "model": info.model_dump() if info is not None else None,
            "policy": policy.model_dump() if policy is not None else None,
        },
    )
st.caption(
    "Scores and development-policy recommendations are not correctness guarantees or out-of-domain detection. No routing or external action is executed. A human remains responsible."
)
