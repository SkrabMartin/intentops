"""Benchmark math and failure transparency with fictional local HTTP responses."""

import json
import math

import httpx
import pytest

from app.schemas import ModelInfoResponse, PolicyInfoResponse
from scripts import benchmark_serving as bench


def test_latency_percentiles_throughput_and_excluded_warmup(monkeypatch):
    model = ModelInfoResponse()
    policy = PolicyInfoResponse()
    calls = []
    monkeypatch.setattr(bench, "_invoke", lambda *a, **k: calls.append(1))
    times = iter([0.0, 0.0, 0.01, 0.01, 0.03, 0.03, 0.06, 0.06])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(times))
    result = bench.measure(
        None, size=10, batch=True, samples=3, warmup=2, model=model, policy=policy
    )
    assert len(calls) == 5 and result["validated_samples"] == 3
    assert result["p50_ms"] == pytest.approx(20)
    assert result["p95_ms"] == pytest.approx(29)
    assert result["p99_ms"] == pytest.approx(29.8)
    assert result["requests_per_second"] == pytest.approx(50)
    assert result["messages_per_second"] == pytest.approx(500)


def local_transport(monkeypatch, *, no_model=False, malformed=False):
    model = ModelInfoResponse(
        status="loaded",
        model_ready=True,
        model_version="fictional-v1",
        classifier_type="TfidfVectorizer + LogisticRegression",
        category_count=4,
        reason=None,
    )
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/ready":
            return httpx.Response(503 if no_model else 200, json=model.model_dump())
        if request.url.path == "/policy/info":
            return httpx.Response(200, json=PolicyInfoResponse().model_dump())
        body = {
            "intent": "fictional",
            "model_version": "fictional-v1",
            "top_predictions": [
                {"intent": "fictional", "score": 0.7},
                {"intent": "second", "score": 0.2},
                {"intent": "third", "score": 0.1},
            ],
        }
        if malformed:
            body = {"private": "must-not-echo"}
        if request.url.path == "/predict/batch":
            body = {"predictions": [body] * len(json.loads(request.content)["messages"])}
        return httpx.Response(200, json=body)

    original = httpx.Client
    monkeypatch.setattr(
        bench.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    return calls


def test_benchmark_validates_real_http_payloads_and_case_sizes(monkeypatch):
    calls = local_transport(monkeypatch)
    report = bench.benchmark(
        "http://127.0.0.1:18000", mode="native", single_requests=2, batch_repeats=2, warmup=1
    )
    assert len(calls) == 17  # 2 discovery + 5*(1 warmup + 2 measured)
    assert [c["batch_size"] for c in report["cases"]] == [1, 1, 10, 50, 100]
    assert all(c["validated_samples"] == 2 and c["errors"] == {} for c in report["cases"])
    assert all(math.isfinite(c["p99_ms"]) for c in report["cases"])
    assert report["concurrency"] == 1 and "not pure model compute" in report["measurement"]


def test_benchmark_no_model_fails_before_measurement(monkeypatch, capsys):
    calls = local_transport(monkeypatch, no_model=True)
    assert bench.main(["--mode", "native"]) == 1
    assert calls == ["/ready"] and "No measurements substituted" in capsys.readouterr().err


def test_benchmark_malformed_result_records_error_and_stops(monkeypatch, capsys):
    local_transport(monkeypatch, malformed=True)
    assert bench.main(["--mode", "native", "--warmup", "0"]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["cases"][0]["validated_samples"] == 0
    assert report["cases"][0]["errors"] == {"invalid_response": 1}
    assert len(report["cases"]) == 1 and "must-not-echo" not in output


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "http://127.0.0.1/path",
        "http://user:password@127.0.0.1",
    ],
)
def test_benchmark_external_api_rejected_without_connection(url):
    with pytest.raises(ValueError):
        bench.benchmark(url, mode="native")


def test_benchmark_counts_bounded():
    with pytest.raises(ValueError):
        bench.benchmark("http://127.0.0.1", mode="native", single_requests=2001)
