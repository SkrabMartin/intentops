"""Phase-0 dashboard; message text stays in the local browser session."""

import os

import httpx
import streamlit as st

st.set_page_config(page_title="IntentOps", page_icon="💬")
st.title("IntentOps")
st.caption("Open-source, local-first customer support intent classification")
st.warning("Phase 0 — Model not trained")
st.write("Classification will be available in a later phase. No predictions are generated yet.")
st.text_area("Future customer message", max_chars=5000, placeholder="Use a fictional example.")
st.button("Classify (available in a later phase)", disabled=True)
st.caption("Message text is not submitted to the API or saved to disk.")

base_url = os.environ.get("INTENTOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.subheader("Local service status")
try:
    # Avoid inherited proxy settings and redirects to other hosts.
    with httpx.Client(base_url=base_url, timeout=2.0, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        info = client.get("/model/info")
        info.raise_for_status()
    st.json({"health": health.json(), "model": info.json()})
except (httpx.HTTPError, ValueError):
    st.info("The local API is unavailable. Start the API, then refresh this page.")
