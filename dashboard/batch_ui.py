"""Streamlit CSV controls with ephemeral, input-free result state."""

import hashlib
import json

import httpx
import streamlit as st

from dashboard.csv_workflow import (
    FICTIONAL_CSV,
    MAX_CSV_BYTES,
    CSVError,
    CSVResult,
    classify_rows,
    export_csv,
    parse_csv,
    result_records,
)

RESULT_KEY = "intentops_csv_results"


def render_csv(base_url: str, *, ready: bool, identity: dict) -> None:
    st.subheader("Classify a CSV locally")
    st.caption(
        "UTF-8 / BOM CSV with exactly one text column. Up to 1 MiB and 200 data rows. No source text is included in the results download."
    )
    st.download_button(
        "Download fictional CSV",
        FICTIONAL_CSV,
        file_name="intentops-fictional.csv",
        mime="text/csv",
        on_click="ignore",
    )
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        max_upload_size=1,
        accept_multiple_files=False,
        disabled=not ready,
    )
    if uploaded is None:
        st.session_state.pop(RESULT_KEY, None)
        st.button("Classify CSV", disabled=True)
        return
    try:
        if uploaded.size > MAX_CSV_BYTES:
            raise CSVError("Upload a CSV no larger than 1 MiB.")
        uploaded.seek(0)
        content = uploaded.read(MAX_CSV_BYTES + 1)
        rows = parse_csv(content)
    except CSVError as exc:
        st.session_state.pop(RESULT_KEY, None)
        st.error(str(exc))  # Only fixed parser messages, never file names/contents.
        st.button("Classify CSV", disabled=True)
        return
    fingerprint = hashlib.sha256(
        content + json.dumps({"api": base_url, "ready": ready, **identity}, sort_keys=True).encode()
    ).hexdigest()
    previous = st.session_state.get(RESULT_KEY)
    if previous and previous["fingerprint"] != fingerprint:
        st.session_state.pop(RESULT_KEY, None)
    valid = sum(row.text is not None for row in rows)
    st.caption(
        f"{len(rows)} rows · {valid} valid · {len(rows) - valid} invalid. Invalid text rows are kept as row errors; structural CSV errors reject the file."
    )
    if st.button("Classify CSV", disabled=not ready or not valid):
        with httpx.Client(
            base_url=base_url, timeout=30.0, trust_env=False, follow_redirects=False
        ) as client:
            results = classify_rows(rows, client)
        st.session_state[RESULT_KEY] = {
            "fingerprint": fingerprint,
            "records": result_records(results),
            "csv": export_csv(results),
        }
    elif not valid:
        results = [CSVResult(row.row_index, "invalid_text") for row in rows]
        st.session_state[RESULT_KEY] = {
            "fingerprint": fingerprint,
            "records": result_records(results),
            "csv": export_csv(results),
        }
    saved = st.session_state.get(RESULT_KEY)
    if saved:
        records = saved["records"]
        classified = sum(row["status"] == "classified" for row in records)
        st.success(f"CSV results: {classified} classified, {len(records) - classified} row errors")
        if classified != len(records):
            st.warning(
                "See Status for each row. After an API failure, remaining rows are marked and no further chunks are sent."
            )
        st.dataframe(records, hide_index=True)
        st.download_button(
            "Download results CSV",
            saved["csv"],
            file_name="intentops-results.csv",
            mime="text/csv",
            on_click="ignore",
        )
    st.caption(
        "Upload contents stay transient in this session. Changing the file or model/policy status clears previous results. Classifications are recommendations; no ticket is routed."
    )
