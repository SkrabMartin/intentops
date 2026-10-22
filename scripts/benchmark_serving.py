"""Sequential local HTTP benchmark. Prints measurements only; redirect to ignored data/."""

import argparse
import json
import os
import platform
import sys
import time

import httpx
import numpy as np

from app.local_http import local_api_url
from app.schemas import (
    BatchPredictionResponse,
    ModelInfoResponse,
    PolicyInfoResponse,
    PredictionResponse,
)

MESSAGES = (
    "When will my new card arrive?",
    "I was charged twice for the same card payment.",
    "Hello, please help.",
)


class BenchmarkError(RuntimeError):
    """Fixed diagnostics, no server exception payload or input bodies."""


def _invoke(
    client, size: int, *, batch: bool, model: ModelInfoResponse, policy: PolicyInfoResponse
):
    messages = [MESSAGES[i % len(MESSAGES)] for i in range(size)]
    response = client.post(
        "/predict/batch" if batch else "/predict",
        json={"messages": messages} if batch else {"text": messages[0]},
    )
    response.raise_for_status()
    results = (
        BatchPredictionResponse.model_validate(response.json()).predictions
        if batch
        else [PredictionResponse.model_validate(response.json())]
    )
    if len(results) != size or any(
        (r.model_version, r.policy_status, r.policy_version)
        != (model.model_version, policy.status, policy.policy_version)
        for r in results
    ):
        raise BenchmarkError("Response identity or count changed")


def measure(client, *, size: int, batch: bool, samples: int, warmup: int, model, policy):
    for _ in range(warmup):
        _invoke(client, size, batch=batch, model=model, policy=policy)
    latency = []
    errors = {}
    start = time.perf_counter()
    for _ in range(samples):
        before = time.perf_counter()
        try:
            _invoke(client, size, batch=batch, model=model, policy=policy)
        except (httpx.HTTPError, ValueError, BenchmarkError) as exc:
            code = "http_error" if isinstance(exc, httpx.HTTPError) else "invalid_response"
            errors[code] = errors.get(code, 0) + 1
            break  # Do not flood an unhealthy service or invent remaining samples.
        latency.append(time.perf_counter() - before)
    elapsed = time.perf_counter() - start
    percentiles = np.percentile(latency, [50, 95, 99]) * 1000 if latency else [None] * 3
    return {
        "endpoint": "/predict/batch" if batch else "/predict",
        "batch_size": size,
        "requested_samples": samples,
        "validated_samples": len(latency),
        "attempted_samples": len(latency) + sum(errors.values()),
        "errors": errors,
        "p50_ms": float(percentiles[0]) if latency else None,
        "p95_ms": float(percentiles[1]) if latency else None,
        "p99_ms": float(percentiles[2]) if latency else None,
        "elapsed_seconds": elapsed,
        "requests_per_second": len(latency) / elapsed,
        "messages_per_second": len(latency) * size / elapsed,
    }


def benchmark(base_url: str, *, mode: str, single_requests=200, batch_repeats=20, warmup=5) -> dict:
    if not 1 <= single_requests <= 2000 or not 1 <= batch_repeats <= 100 or not 0 <= warmup <= 20:
        raise ValueError("Benchmark sample counts outside documented bounds")
    base_url = local_api_url(base_url)
    with httpx.Client(
        base_url=base_url, timeout=30.0, trust_env=False, follow_redirects=False
    ) as client:
        response = client.get("/ready")
        response.raise_for_status()
        model = ModelInfoResponse.model_validate(response.json())
        response = client.get("/policy/info")
        response.raise_for_status()
        policy = PolicyInfoResponse.model_validate(response.json())
        if not model.model_ready:
            raise BenchmarkError("A real ready model is required")
        cases = []
        for batch, size, count in [
            (False, 1, single_requests),
            *[(True, size, batch_repeats) for size in (1, 10, 50, 100)],
        ]:
            case = measure(
                client,
                size=size,
                batch=batch,
                samples=count,
                warmup=warmup,
                model=model,
                policy=policy,
            )
            cases.append(case)
            if case["errors"]:
                break
    return {
        "measurement": "HTTP round trip plus status, JSON and typed payload validation; not pure model compute",
        "mode": mode,
        "concurrency": 1,
        "warmup_per_case_excluded": warmup,
        "configured_single_samples": single_requests,
        "configured_batch_repetitions": batch_repeats,
        "percentiles": "numpy linear interpolation, successful validated requests only",
        "throughput": "validated successes / wall time, including any failed attempt; no SLA",
        "client_environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "not reported",
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "httpx": httpx.__version__,
        },
        "server_python": "Not exposed by API; record operator-verified native/container environment separately",
        "model": model.model_dump(),
        "policy": policy.model_dump(),
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--mode", choices=["native", "docker"], required=True)
    parser.add_argument("--single-requests", type=int, default=200)
    parser.add_argument("--batch-repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = benchmark(
            args.url,
            mode=args.mode,
            single_requests=args.single_requests,
            batch_repeats=args.batch_repeats,
            warmup=args.warmup,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if any(c["errors"] for c in result["cases"]) else 0
    except (httpx.HTTPError, ValueError, BenchmarkError):
        print(
            "Benchmark failed: use a ready, compatible local API and valid bounded configuration. No measurements substituted.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
