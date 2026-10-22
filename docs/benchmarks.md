# Local serving benchmark

Run a real compatible API first; the benchmark refuses no-model and remote
origins, validates every response and never substitutes a dummy model. Only
three fixed fictional inputs are used. No dataset, model file or policy file is
opened by the benchmark client. No fitting, tuning or audit is performed.

From the installed project environment (PowerShell):

```powershell
python scripts/benchmark_serving.py --mode native --url http://127.0.0.1:18000 > data/benchmark-native.json
python scripts/benchmark_serving.py --mode docker --url http://127.0.0.1:18000 > data/benchmark-docker.json
```

Use the command matching the running deployment. Keep output under ignored data/;
create that directory explicitly if absent. --mode labels operator deployment,
not an automatically detected server platform. Metadata records actual model and
policy version/state. Exact server Python is not exposed by the public API;
record independently verified server/container versions alongside measurements.

Default concurrency 1, persistent HTTP client, 200 single requests plus20 requests
each at batch sizes1,10,50,100. Five warmup requests per case are excluded from
all latency/throughput statistics. Inputs are deterministic; timing is not.
Bounds: single requests 1..2000, batch repeats 1..100, warmup 0..20. Batch requests
use the real bounded endpoint. Timeout30 s; proxy inheritance and redirects off.

Each measurement covers HTTP round trip, status, JSON parsing and strict typed
payload/count/identity validation. It is **not pure model compute latency**.
P50/P95/P99 use NumPy linear interpolation over successful validated requests;
small batch sample sizes make tail percentiles imprecise. Throughput counts
validated requests and messages per elapsed wall time, including any failure.
Errors are counted separately, stop further workload and cause nonzero exit.
Warmup/discovery failures also fail explicitly with no invented measurements.

JSON records client OS/Python/processor/machine/logical CPUs/httpx, deployment
label, model/policy metadata, all sample counts, successful counts, errors,
percentiles, wall time and throughput. Host-specific reports remain local.
No SLA, production certification, fairness claim or tuning from these timings.
Windows native and Linux Docker observations are separate configurations; do
not treat differences as controlled platform comparisons. Frozen policy caveat:
10.13% audit upper error bound exceeded the 10% demonstration budget.

## Observed native execution (2026-10-22)

Policy-enabled Windows native server, Python 3.12.13, existing model
`tfidf-lr-7a1a49c4228bb52f`, frozen policy `review-d1995987cf7ca46e`.
Client environment: {"os": "Windows-11-10.0.26200-SP0", "python": "3.12.13", "processor": "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel", "machine": "AMD64", "logical_cpus": 12, "httpx": "0.28.1"}.
Concurrency1; 200 single samples and20 per batch,5 excluded warmups per case; all
responses validated, zero errors. Exact server environment was separately checked.

| HTTP case | Samples | P50 ms | P95 ms | P99 ms | Requests/s | Messages/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single | 200 | 20.50 | 28.74 | 34.58 | 46.95 | 46.95 |
| batch 1 | 20 | 20.71 | 24.43 | 24.84 | 49.01 | 49.01 |
| batch 10 | 20 | 22.84 | 27.93 | 28.41 | 43.52 | 435.22 |
| batch 50 | 20 | 28.18 | 34.70 | 45.21 | 34.12 | 1705.79 |
| batch 100 | 20 | 34.07 | 41.26 | 45.16 | 28.16 | 2815.64 |

These are observations from one sequential run, including typed client validation,
not a controlled hardware/platform comparison or guaranteed capacity. Small20-sample
batch percentiles are descriptive. The separate Docker run is reported below; the configurations are not a controlled comparison.


## Observed Docker execution (2026-10-22)

Policy-enabled Linux/WSL2 server, Python 3.12.15, preserved model
`tfidf-lr-a7783428fd040ecb`, frozen policy `review-ee4f34ee8964058c`.
Server environment independently checked: {"os": "Linux", "kernel": "6.18.54.1-microsoft-standard-WSL2", "python": "3.12.15", "logical_cpus": 12}.
The benchmark client was the same Windows Python 3.12.13 /12 logical CPU machine
recorded above, calling the published loopback port18000. Concurrency1,200 single
samples and20 per batch,5 excluded warmups per case, all responses validated,zero errors.

| HTTP case | Samples | P50 ms | P95 ms | P99 ms | Requests/s | Messages/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single | 200 | 15.92 | 22.35 | 26.81 | 59.94 | 59.94 |
| batch 1 | 20 | 17.05 | 23.21 | 25.19 | 56.80 | 56.80 |
| batch 10 | 20 | 16.63 | 28.76 | 31.09 | 54.19 | 541.89 |
| batch 50 | 20 | 23.53 | 30.70 | 40.74 | 41.21 | 2060.32 |
| batch 100 | 20 | 27.84 | 37.79 | 38.25 | 35.59 | 3558.52 |

The two observations involve different server OS/Python environments, virtualization
and uncontrolled background load. They are neither a Docker-vs-native causal claim
nor an SLA. Larger batches amortized HTTP overhead in these specific runs; timings
are not evidence of improved classification accuracy or policy safety.
