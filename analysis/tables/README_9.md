# Notebook 9: FastAPI Serving Benchmark

Hardware: AMD EPYC 7763 CPU (FastAPI server runs ONNX Runtime on CPU). NVIDIA A100 80GB available but not used by FastAPI.

Server: FastAPI + ONNX Runtime (`CPUExecutionProvider`), single uvicorn worker.

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `http://fastapi_server:8000/health` | GET | Health check |
| `http://fastapi_server:8000/predict/global` | POST | Single global MLP prediction |
| `http://fastapi_server:8000/predict/global/batch` | POST | Batch global MLP prediction |
| `http://fastapi_server:8000/predict/personalized` | POST | Single personalized MLP prediction |
| `http://fastapi_server:8000/predict/personalized/batch` | POST | Batch personalized MLP prediction |

---

## Global MLP — Sequential Benchmarks (measured)

| Scenario | Endpoint | Median (ms) | p95 (ms) | p99 (ms) | Throughput |
|----------|----------|----------:|--------:|--------:|---------:|
| Single request | `POST /predict/global` | 5.36 | 6.27 | 7.32 | 182.05 req/s |
| Batch=32 | `POST /predict/global/batch` | 42.03 | 46.57 | — | 745.41 samples/s |

## Global MLP — Concurrent Benchmarks (measured, n=500 per level)

| Concurrency | Endpoint | Median (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | CPU util avg | RAM avg (GB) |
|------------|----------|----------:|--------:|--------:|------------------:|----------:|----------:|
| 1 | `POST /predict/global` | 2.35 | 2.69 | 2.82 | 396.35 | 1.9% | 5.76 |
| 4 | `POST /predict/global` | 9.46 | 13.21 | 15.63 | 378.93 | 2.1% | 5.72 |
| 8 | `POST /predict/global` | 16.79 | 24.89 | 28.51 | 419.88 | 2.1% | 5.70 |
| 16 | `POST /predict/global` | 30.36 | 46.45 | 54.93 | 462.76 | 3.2% | 5.67 |

---

## Personalized MLP — Sequential Benchmarks (estimated)

The personalized cells were not executed in this notebook run. The estimates below are derived from the measured global results and the known compute overhead ratio from NB6 (personalized raw ONNX is ~1.3x slower single, ~3.5x slower batch vs global). Since FastAPI overhead (~5.3 ms per request) dominates over model compute (~0.03–0.04 ms), the personalized numbers are nearly identical to global for single requests.

| Scenario | Endpoint | Est. Median (ms) | Est. p95 (ms) | Est. p99 (ms) | Est. Throughput |
|----------|----------|----------------:|-------------:|-------------:|--------------:|
| Single request | `POST /predict/personalized` | ~5.4 | ~6.3 | ~7.4 | ~180 req/s |
| Batch=32 | `POST /predict/personalized/batch` | ~42.3 | ~47.0 | — | ~740 samples/s |

## Personalized MLP — Concurrent Benchmarks (estimated, n=500 per level)

| Concurrency | Endpoint | Est. Median (ms) | Est. p95 (ms) | Est. p99 (ms) | Est. Throughput (req/s) |
|------------|----------|----------------:|-------------:|-------------:|----------------------:|
| 1 | `POST /predict/personalized` | ~2.4 | ~2.7 | ~2.9 | ~390 |
| 4 | `POST /predict/personalized` | ~9.5 | ~13.3 | ~15.7 | ~375 |
| 8 | `POST /predict/personalized` | ~16.9 | ~25.0 | ~28.6 | ~415 |
| 16 | `POST /predict/personalized` | ~30.5 | ~46.6 | ~55.1 | ~458 |

**Estimation methodology:** FastAPI single-request overhead is ~5.33 ms (measured global latency 5.36 ms minus raw ONNX compute 0.03 ms). Personalized raw ONNX compute is 0.04 ms (from NB6). Total = 5.33 + 0.04 ≈ 5.37 ms. For batch=32, global overhead was ~41.9 ms (42.03 ms minus 32/322K = 0.1 ms compute); personalized compute is 32/93K = 0.34 ms; total ≈ 42.3 ms. Concurrent benchmarks are HTTP-bound (not compute-bound), so personalized numbers track global within ~1%.

---

## Key Observations

- Single-request latency is ~5.4 ms for both models — dominated by HTTP overhead (JSON serialization of 768 floats), not model compute (~0.03–0.04 ms).
- Batch=32 latency is ~42 ms — the JSON payload is 32×768 = 24,576 floats, so serialization dominates.
- Concurrent throughput peaks at ~463 req/s (concurrency=16), but per-request latency degrades to ~30 ms median. The single uvicorn worker serializes all requests.
- GPU utilization is 0% — FastAPI uses `CPUExecutionProvider` only.
- CPU utilization stays under 3.2% even at concurrency=16 — the bottleneck is I/O (HTTP round-trips), not compute.
- The personalized model adds negligible overhead vs global in FastAPI serving because HTTP serialization dominates model compute by ~100x.
