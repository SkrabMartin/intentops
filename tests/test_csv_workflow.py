"""Fictional CSV structure, row outcomes, bounded calls, privacy and actual UI upload."""

import csv
import io
import json

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from test_dashboard import SCRIPT, setup_http

from dashboard.batch_ui import RESULT_KEY
from dashboard.csv_workflow import (
    FICTIONAL_CSV,
    MAX_CSV_BYTES,
    CSVError,
    CSVResult,
    classify_rows,
    export_csv,
    parse_csv,
)


def prediction(intent="fictional_intent"):
    return {
        "intent": intent,
        "model_version": "fictional-v1",
        "top_predictions": [
            {"intent": intent, "score": 0.7},
            {"intent": "fictional_second", "score": 0.2},
            {"intent": "fictional_third", "score": 0.1},
        ],
    }


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"text\n",
        b"other\nprivate\n",
        b"text,text\na,b\n",
        b"id,text\n1,private\n",
        b"text\na,b\n",
        b'text\n"unclosed',
        b"text\n\xff",
        b"text\na\x00b\n",
        b"text\n\n",
        b"text\n" + b"x\n" * 201,
        b"x" * (MAX_CSV_BYTES + 1),
    ],
    ids=lambda content: f"csv-{len(content)}-bytes",
)
def test_bad_structure_or_limits_reject_safely(content):
    with pytest.raises(CSVError) as error:
        parse_csv(content)
    assert "private" not in str(error.value) and "unclosed" not in str(error.value)


def test_bom_quotes_newlines_and_mixed_validation():
    assert len(parse_csv(FICTIONAL_CSV)) == 3
    assert all(r.text is not None for r in parse_csv(FICTIONAL_CSV))
    rows = parse_csv(b'\xef\xbb\xbftext\n"fictional, comma"\n"fictional\nnewline"\n""\n"   "\n')
    assert [r.row_index for r in rows] == [1, 2, 3, 4]
    assert rows[0].text == "fictional, comma" and rows[1].text == "fictional\nnewline"
    assert rows[2].text is None and rows[3].text is None
    long = parse_csv(("text\n" + "x" * 5001 + "\n").encode())
    assert long[0].text is None


def test_chunking_order_and_no_text_or_formula_in_export():
    content = ("text\n" + "fictional valid row\n" * 199 + '""\n').encode()
    rows = parse_csv(content)
    calls = []

    def handle(request):
        messages = json.loads(request.content)["messages"]
        calls.append(len(messages))
        return httpx.Response(200, json={"predictions": [prediction() for _ in messages]})

    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(handle)) as client:
        results = classify_rows(rows, client)
    assert calls == [100, 99]
    assert [r.row_index for r in results] == list(range(1, 201))
    assert results[-1].status == "invalid_text"
    output = export_csv(results)
    parsed = list(csv.DictReader(io.StringIO(output.decode())))
    assert len(parsed) == 200 and "text" not in parsed[0]
    assert "fictional valid row" not in output.decode()
    for unsafe in ['=HYPERLINK("x")', "+cmd", "-cmd", "@SUM(1)", "\t=1", "\r=1", " \ufeff=1"]:
        exported = export_csv([CSVResult(1, "classified", intent=unsafe, model_version=unsafe)])
        record = next(csv.DictReader(io.StringIO(exported.decode())))
        assert record["intent"].startswith("'") and record["model_version"].startswith("'")


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("timeout", "api_timeout"),
        ("503", "api_unavailable"),
        ("down", "api_unavailable"),
        ("500", "api_error"),
        ("malformed", "invalid_response"),
        ("count", "invalid_response"),
    ],
)
def test_partial_failure_marks_remaining_rows_and_stops_calls(failure, expected):
    rows = parse_csv(("text\n" + "fictional\n" * 200).encode())
    calls = []

    def handle(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json={"predictions": [prediction()] * 100})
        if failure == "timeout":
            raise httpx.ReadTimeout("private-input-path")
        if failure == "down":
            raise httpx.ConnectError("private-input-path")
        if failure in ("503", "500"):
            return httpx.Response(int(failure), json={"private": "must-not-echo"})
        return httpx.Response(
            200,
            json={"predictions": []} if failure == "malformed" else {"predictions": [prediction()]},
        )

    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(handle)) as client:
        results = classify_rows(rows, client)
    assert len(calls) == 2 and len(results) == 200
    assert all(r.status == "classified" for r in results[:100])
    assert all(r.status == expected for r in results[100:])
    assert b"private-input-path" not in export_csv(results) and b"must-not-echo" not in export_csv(
        results
    )


def setup_batch_http(monkeypatch, *, ready=True, failure=None):
    setup_http(monkeypatch, ready=ready)
    calls = []

    def post(client, path, *, json):
        assert path == "/predict/batch"
        calls.append(json)
        if failure == "timeout":
            raise httpx.ReadTimeout("private timeout detail")
        if failure == "503":
            return httpx.Response(503, request=httpx.Request("POST", "http://api" + path))
        return httpx.Response(
            200,
            json={"predictions": [prediction()] * len(json["messages"])},
            request=httpx.Request("POST", "http://api" + path),
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    return calls


def upload(view, content):
    return view.file_uploader[0].set_value(("fictional.csv", content, "text/csv")).run(timeout=25)


def csv_button(view):
    return next(b for b in view.button if b.label == "Classify CSV")


def test_actual_upload_classify_download_mixed_rows_and_no_implicit_calls(monkeypatch):
    calls = setup_batch_http(monkeypatch)
    view = AppTest.from_file(SCRIPT).run(timeout=25)
    assert not view.exception and [t.label for t in view.tabs] == ["Single Message", "Batch CSV"]
    content = b'text\nfictional private first\n""\nfictional private third\n'
    view = upload(view, content)
    assert calls == [] and not csv_button(view).disabled
    csv_button(view).click().run(timeout=25)
    assert not view.exception and len(calls) == 1
    records = view.session_state[RESULT_KEY]["records"]
    assert [r["status"] for r in records] == ["classified", "invalid_text", "classified"]
    assert view.download_button[-1].label == "Download results CSV"
    view.download_button[-1].click().run(timeout=25)
    assert len(calls) == 1 and not view.exception
    output = view.session_state[RESULT_KEY]["csv"]
    parsed = list(csv.DictReader(io.StringIO(output.decode())))
    assert [r["row_index"] for r in parsed] == ["1", "2", "3"]
    assert b"fictional private first" not in output and b"fictional private third" not in output
    assert "text" not in parsed[0]
    upload(view, b"text\nchanged fictional file\n")
    assert RESULT_KEY not in view.session_state and len(calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        b"bad\nprivate\n",
        b'text\n"unterminated',
        b"text\n" + b"x\n" * 201,
        b"x" * (MAX_CSV_BYTES + 1),
    ],
    ids=lambda content: f"csv-{len(content)}-bytes",
)
def test_ui_rejects_invalid_upload_without_post(monkeypatch, content):
    calls = setup_batch_http(monkeypatch)
    view = upload(AppTest.from_file(SCRIPT).run(timeout=25), content)
    assert not view.exception and calls == [] and csv_button(view).disabled and view.error


@pytest.mark.parametrize("failure,status", [("timeout", "api_timeout"), ("503", "api_unavailable")])
def test_ui_api_failure_export_is_honest(monkeypatch, failure, status):
    calls = setup_batch_http(monkeypatch, failure=failure)
    view = upload(AppTest.from_file(SCRIPT).run(timeout=25), b"text\nfictional\n")
    csv_button(view).click().run(timeout=25)
    assert not view.exception and len(calls) == 1
    assert view.session_state[RESULT_KEY]["records"][0]["status"] == status
    assert all("private timeout detail" not in e.value for e in view.error)


def test_no_model_csv_controls_disabled_and_results_clear(monkeypatch):
    calls = setup_batch_http(monkeypatch)
    view = upload(AppTest.from_file(SCRIPT).run(timeout=25), b"text\nfictional\n")
    csv_button(view).click().run(timeout=25)
    assert RESULT_KEY in view.session_state
    setup_batch_http(monkeypatch, ready=False)
    view.run(timeout=25)
    assert not view.exception and csv_button(view).disabled and view.file_uploader[0].disabled
    assert RESULT_KEY not in view.session_state and len(calls) == 1
