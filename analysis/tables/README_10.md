# Notebook 10: Triton Inference Server Benchmark

Hardware: NVIDIA A100 80GB + AMD EPYC 7763 CPU (Chameleon Cloud). Triton 24.10 with ONNX Runtime backend, 1 GPU instance (baseline), FP32 ONNX models.

Triton config: `backend: "onnxruntime"`, `max_batch_size: 64`, 1 instance on GPU 0 (baseline).

---

## Triton HTTP API Endpoints

| Endpoint | URL |
|----------|-----|
| Health | `http://triton_server:8000/v2/health/ready` |
| Server metadata | `http://triton_server:8000/v2` |
| Global MLP metadata | `http://triton_server:8000/v2/models/flickr_global` |
| Global MLP inference | `http://triton_server:8000/v2/models/flickr_global/infer` |
| Global MLP stats | `http://triton_server:8000/v2/models/flickr_global/versions/1/stats` |
| Personalized MLP metadata | `http://triton_server:8000/v2/models/flickr_personalized` |
| Personalized MLP inference | `http://triton_server:8000/v2/models/flickr_personalized/infer` |
| Personalized MLP stats | `http://triton_server:8000/v2/models/flickr_personalized/versions/1/stats` |
| Prometheus metrics | `http://triton_server:8002/metrics` |

---

## Part 2 & 3: Python Client Benchmarks (1 instance, no dynamic batching)

### Sequential Benchmarks

| Model | Endpoint | Scenario | Median (ms) | p95 (ms) | p99 (ms) | Throughput |
|-------|----------|----------|----------:|--------:|--------:|---------:|
| Global | `POST .../flickr_global/infer` | Single | 0.66 | 0.69 | 0.72 | 1,503 infer/s |
| Global | `POST .../flickr_global/infer` | Batch=32 | 0.95 | — | — | 33,448 samples/s |
| Personalized | `POST .../flickr_personalized/infer` | Single | 0.87 | 0.89 | 0.91 | 1,146 infer/s |
| Personalized | `POST .../flickr_personalized/infer` | Batch=32 | 0.61 | — | — | 50,629 samples/s |

### Server-side GPU Metrics

| Metric | Value |
|--------|------:|
| GPU utilization | 0.0% |
| GPU memory used | 539 MB |
| GPU memory total | 81,920 MB (80 GB) |

---

## Part 4: `perf_analyzer` — Concurrency Sweep (1 instance, Global MLP)

| Concurrency | Endpoint | Throughput (infer/s) | Avg Latency (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Queue (µs) | Compute Infer (µs) |
|------------|----------|-------------------:|----------------:|--------:|--------:|--------:|---------:|------------------:|
| 1 | `POST .../flickr_global/infer` | 1,874 | 524 | 524 | 546 | 574 | 24 | 203 |
| 8 | `POST .../flickr_global/infer` | 7,059 | 1,125 | 1,117 | 1,176 | 1,688 | 835 | 110 |
| 16 | `POST .../flickr_global/infer` | 7,052 | 2,263 | 2,250 | 2,361 | 3,238 | 1,958 | 110 |

Throughput saturates at concurrency=8 (7,059/s). Going to 16 doubles latency with no throughput gain — queue delay dominates (1,958 µs).

## Part 4: `perf_analyzer` — Batch Size Sweep (1 instance, concurrency=1, Global MLP)

| Batch Size | Endpoint | Throughput (infer/s) | Avg Latency (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Queue (µs) | Compute Infer (µs) |
|-----------|----------|-------------------:|----------------:|--------:|--------:|--------:|---------:|------------------:|
| 1 | `POST .../flickr_global/infer` | 1,814 | 538 | 536 | 561 | 592 | 25 | 205 |
| 8 | `POST .../flickr_global/infer` | 13,103 | 604 | 605 | 622 | 647 | 24 | 225 |
| 16 | `POST .../flickr_global/infer` | 23,478 | 672 | 673 | 693 | 731 | 25 | 229 |
| 32 | `POST .../flickr_global/infer` | 39,749 | 795 | 791 | 827 | 879 | 26 | 230 |
| 64 | `POST .../flickr_global/infer` | **68,809** | 920 | 885 | 1,054 | 1,214 | 23 | 197 |

Throughput scales linearly with batch size. b=64 achieves 68.8K infer/s at only 920 µs latency.

---

## Part 5: Scaling — 2 Instances (Global MLP)

| Instances | Concurrency | Endpoint | Throughput (infer/s) | Avg Latency (µs) | Queue (µs) | Compute Infer (µs) |
|----------|------------|----------|-------------------:|----------------:|---------:|------------------:|
| 1 | 1 | `POST .../flickr_global/infer` | 1,874 | 524 | 24 | 203 |
| 2 | 1 | `POST .../flickr_global/infer` | 1,788 | 549 | 27 | 199 |
| 1 | 8 | `POST .../flickr_global/infer` | 7,059 | 1,125 | 835 | 110 |
| **2** | **8** | `POST .../flickr_global/infer` | **9,717** | **810** | **431** | **159** |
| 1 | 16 | `POST .../flickr_global/infer` | 7,052 | 2,263 | 1,958 | 110 |
| **2** | **16** | `POST .../flickr_global/infer` | **9,656** | **1,647** | **1,248** | **159** |

At concurrency=8: 2 instances give +38% throughput (9,717 vs 7,059) and -49% queue delay (431 vs 835 µs).

---

## Part 6: Dynamic Batching (1 instance, Global MLP)

Config: `preferred_batch_size: [4, 8, 16]`, `max_queue_delay_microseconds: 100`. Poisson arrival distribution.

| Scenario | Endpoint | Rate (req/s) | Throughput (infer/s) | Avg Latency (µs) | Queue (µs) | Compute Infer (µs) | Executions | Inferences |
|----------|----------|------------:|-------------------:|----------------:|---------:|------------------:|---------:|---------:|
| No batching, rate=200 | `POST .../flickr_global/infer` | 200 | 201 | 779 | 82 | 236 | 3,627 | 3,627 |
| Batching ON, rate=200 | `POST .../flickr_global/infer` | 200 | 201 | 924 | 222 | 240 | 3,506 | 3,627 |
| Batching ON, rate=500 | `POST .../flickr_global/infer` | 500 | 500 | 829 | 185 | 206 | 8,339 | 9,007 |

At rate=200, dynamic batching adds ~145 µs latency (queue delay increases from 82→222 µs) but reduces executions from 3,627→3,506 (batching ~3.5% of requests). At rate=500, batching is more effective: 8,339 executions for 9,007 inferences (1.08x batch ratio), with lower latency than rate=200 without batching.

---

## Summary — Python Client (all models)

| Scenario | Endpoint | Median (ms) | p95 (ms) | Throughput |
|----------|----------|----------:|--------:|---------:|
| Global single | `POST .../flickr_global/infer` | 0.66 | 0.69 | 1,503 infer/s |
| Global batch=32 | `POST .../flickr_global/infer` | 0.95 | — | 33,448 samples/s |
| Personalized single | `POST .../flickr_personalized/infer` | 0.87 | 0.89 | 1,146 infer/s |
| Personalized batch=32 | `POST .../flickr_personalized/infer` | 0.61 | — | 50,629 samples/s |

## Key Observations

- Triton single-sample latency (~0.66 ms) is 8x lower than FastAPI (~5.4 ms) — binary tensor protocol vs JSON serialization.
- Batch throughput at b=64 reaches 68.8K infer/s (perf_analyzer) — the MLP is so fast that Triton overhead is the bottleneck.
- 2 instances provide +38% throughput at concurrency=8 but no benefit at concurrency=1 (no queue to parallelize).
- Dynamic batching adds slight latency at low request rates but becomes effective at higher rates (500 req/s).
- GPU utilization stays at 0% — the MLP is too small to register on nvidia-smi's 0.5s polling interval.
