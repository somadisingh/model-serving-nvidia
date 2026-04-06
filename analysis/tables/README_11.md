# Notebook 11: Production Pipeline Benchmark

Hardware: NVIDIA A100 80GB PCIe + AMD EPYC 7763 CPU (Chameleon Cloud `compute_gigaio` at CHI@UC).

Triton config: 2 GPU instances per model, dynamic batching (`preferred_batch_size: [4, 8, 16, 32, 64, 128]`, `max_queue_delay: 50 µs`), `max_batch_size: 128`.

Production models: Global MLP = Dynamic INT8 (0.47 MB), Personalized MLP = Graph-optimized (1.90 MB).

---

## Triton Endpoint URLs

| Protocol | Endpoint | URL |
|----------|----------|-----|
| HTTP | Health | `http://triton_server:8000/v2/health/ready` |
| HTTP | Server metadata | `http://triton_server:8000/v2` |
| HTTP | Global MLP metadata | `http://triton_server:8000/v2/models/flickr_global` |
| HTTP | Global MLP inference | `http://triton_server:8000/v2/models/flickr_global/infer` |
| HTTP | Global MLP stats | `http://triton_server:8000/v2/models/flickr_global/versions/1/stats` |
| HTTP | Personalized MLP metadata | `http://triton_server:8000/v2/models/flickr_personalized` |
| HTTP | Personalized MLP inference | `http://triton_server:8000/v2/models/flickr_personalized/infer` |
| HTTP | Personalized MLP stats | `http://triton_server:8000/v2/models/flickr_personalized/versions/1/stats` |
| HTTP | Prometheus metrics | `http://triton_server:8002/metrics` |
| gRPC | All models | `triton_server:8001` (binary protobuf over HTTP/2) |

---

## Part 1: CLIP ViT-L/14 — Compiled, Batch Size 128

| Metric | Value |
|--------|------:|
| Throughput | 927.5 FPS |
| Median latency (p50) | 137.94 ms |
| p95 latency | 139.14 ms |
| GPU mem peak (PyTorch) | 1,947 MB |
| GPU util avg | 92.8% |
| GPU util peak | 100.0% |
| GPU mem avg (nvidia-smi) | 3,621 MB |
| CPU util avg | 0.4% |
| RAM used avg | 7.76 GB |

### Production Sizing — ViT Encoding

| Library Size | Batches | Wall Time |
|-------------|--------:|----------:|
| 2,000 | 16 | 2.2s |
| 5,000 | 40 | 5.4s |
| 10,000 | 79 | 10.8s |
| 20,000 | 157 | 21.6s |
| 50,000 | 391 | 53.9s |

---

## Part 2: Global MLP — Dynamic INT8 (CPU, ONNX Runtime)

| Metric | Value |
|--------|------:|
| Model size | 0.47 MB |
| Execution provider | CPUExecutionProvider |
| Single latency (p50) | 0.0825 ms |
| Single latency (p95) | 0.1050 ms |
| Single throughput | 11,038 FPS |
| Batch throughput (b=32) | 111,556 FPS |
| Sample scores (first 5) | 0.621, 0.500, 0.567, 0.573, 0.543 |
| Batch mean / std | 0.543 / 0.087 |

### Production Sizing — Global MLP Scoring

| Library Size | Wall Time |
|-------------|----------:|
| 2,000 | 17.9 ms |
| 5,000 | 44.8 ms |
| 10,000 | 89.6 ms |
| 20,000 | 179.3 ms |
| 50,000 | 448.2 ms |

---

## Part 3: Personalized MLP — Graph-Optimized (CPU, ONNX Runtime)

| Metric | Value |
|--------|------:|
| Model size | 1.90 MB |
| Execution provider | CPUExecutionProvider |
| Inputs | embedding [batch, 768] FP32 + user_idx [batch] INT64 |
| Single latency (p50) | 0.0494 ms |
| Single latency (p95) | 0.0619 ms |
| Single throughput | 16,886 FPS |
| Batch throughput (b=32) | 91,549 FPS |
| Sample scores (first 5, user_idx=0) | 0.587, 0.263, 0.397, 0.423, 0.419 |
| Batch mean / std | 0.398 / 0.118 |

### Production Sizing — Weekly Re-scoring

| Users | Total Images | Wall Time |
|------:|-----------:|----------:|
| 100 | 500,000 | 5.5s |
| 500 | 2,500,000 | 27.3s |
| 1,000 | 5,000,000 | 54.6s |
| 5,000 | 25,000,000 | 273.1s |

---

## Part 4: Triton HTTP Benchmarks (Python Client)

### Sequential Benchmarks

| Model | Endpoint | Scenario | Median (ms) | p95 (ms) | p99 (ms) | Throughput |
|-------|----------|----------|----------:|--------:|--------:|---------:|
| Global | `POST .../flickr_global/infer` | Single | 0.64 | 0.76 | 0.91 | 1,492 infer/s |
| Global | `POST .../flickr_global/infer` | Batch=32 | 0.91 | — | — | 36,752 samples/s |
| Global | `POST .../flickr_global/infer` | Batch=64 | 1.03 | — | — | 62,847 samples/s |
| Personalized | `POST .../flickr_personalized/infer` | Single | 0.68 | 0.72 | 0.86 | 1,483 infer/s |
| Personalized | `POST .../flickr_personalized/infer` | Batch=32 | 0.85 | — | — | 38,202 samples/s |
| Personalized | `POST .../flickr_personalized/infer` | Batch=64 | 1.17 | — | — | 54,729 samples/s |

### Resource Usage (Global single-sample benchmark)

| Metric | Value |
|--------|------:|
| GPU util avg | 0.0% |
| GPU mem | 3,663 MB |
| CPU util avg | 10.5% |
| RAM used avg | 7.22 GB |

---

## Part 4: Triton HTTP `perf_analyzer` (Production Config)

### Global MLP — Concurrency Sweep (many users, single photo upload)

| Concurrency | Endpoint | Throughput (infer/s) | Avg Latency (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Queue (µs) | Compute Infer (µs) |
|------------|----------|-------------------:|----------------:|--------:|--------:|--------:|---------:|------------------:|
| 1 | `POST .../flickr_global/infer` | 1,205 | 821 | 832 | 862 | 902 | 139 | 346 |
| 8 | `POST .../flickr_global/infer` | **13,342** | 594 | 588 | 702 | 777 | 78 | 271 |
| 16 | `POST .../flickr_global/infer` | 13,310 | 1,193 | 1,195 | 1,570 | 1,747 | 295 | 468 |

### Global MLP — Batch Size Sweep (one user, multiple photo uploads)

| Batch Size | Endpoint | Throughput (infer/s) | Avg Latency (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Queue (µs) | Compute Infer (µs) |
|-----------|----------|-------------------:|----------------:|--------:|--------:|--------:|---------:|------------------:|
| 1 | `POST .../flickr_global/infer` | Failed (unstable) | — | — | — | — | — | — |
| 32 | `POST .../flickr_global/infer` | 29,322 | 1,083 | 1,075 | 1,293 | 1,329 | 50 | 537 |
| 64 | `POST .../flickr_global/infer` | **47,980** | 1,326 | 1,320 | 1,395 | 1,550 | 47 | 694 |

### Personalized MLP — Concurrency Sweep (many users, single photo updates)

| Concurrency | Endpoint | Throughput (infer/s) | Avg Latency (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Queue (µs) | Compute Infer (µs) |
|------------|----------|-------------------:|----------------:|--------:|--------:|--------:|---------:|------------------:|
| 1 | `POST .../flickr_personalized/infer` | 1,527 | 641 | 622 | 739 | 763 | 133 | 155 |
| 8 | `POST .../flickr_personalized/infer` | 13,661 | 575 | 564 | 700 | 793 | 89 | 184 |
| 16 | `POST .../flickr_personalized/infer` | **22,085** | 718 | 711 | 893 | 994 | 157 | 217 |

### Personalized MLP — Batch Size Sweep (one user, multiple photo updates)

| Batch Size | Endpoint | Throughput (infer/s) | Avg Latency (µs) | p50 (µs) | p95 (µs) | p99 (µs) | Queue (µs) | Compute Infer (µs) |
|-----------|----------|-------------------:|----------------:|--------:|--------:|--------:|---------:|------------------:|
| 1 | `POST .../flickr_personalized/infer` | 1,447 | 682 | 690 | 738 | 758 | 139 | 196 |
| 32 | `POST .../flickr_personalized/infer` | 29,162 | 1,089 | 1,076 | 1,165 | 1,199 | 49 | 451 |
| 64 | `POST .../flickr_personalized/infer` | **41,708** | 1,526 | 1,529 | 1,592 | 1,631 | 50 | 709 |

---

## Dynamic Batching Statistics (HTTP)

### flickr_global (total inferences: 1,089,260)

| Batch Size | Batches Executed |
|-----------|----------------:|
| 1 | 280,763 |
| 2 | 26,506 |
| 3 | 35,079 |
| 4 | 64,351 |
| 5 | 10,803 |
| 6 | 6,504 |
| 7 | 5,557 |
| 8 | 10,578 |
| 9 | 2,730 |
| 10 | 945 |
| 11 | 492 |
| 12 | 400 |
| 13 | 57 |
| 14 | 1 |
| 32 | 104,004 |
| 64 | 27,291 |

### flickr_personalized (total inferences: 1,241,962)

| Batch Size | Batches Executed |
|-----------|----------------:|
| 1 | 163,891 |
| 2 | 50,359 |
| 3 | 55,583 |
| 4 | 88,351 |
| 5 | 17,318 |
| 6 | 8,494 |
| 7 | 5,811 |
| 8 | 15,419 |
| 9 | 4,641 |
| 10 | 1,733 |
| 11 | 578 |
| 12 | 265 |
| 13 | 34 |
| 14 | 9 |
| 15 | 1 |
| 32 | 33,135 |
| 64 | 53,262 |

---

## Part 5: Triton gRPC Benchmarks (Python Client)

### Sequential Benchmarks

| Model | Protocol | Scenario | Median (ms) | p95 (ms) | p99 (ms) | Throughput |
|-------|----------|----------|----------:|--------:|--------:|---------:|
| Global | gRPC (`triton_server:8001`) | Single | 1.02 | 1.25 | 1.43 | 946 infer/s |
| Global | gRPC (`triton_server:8001`) | Batch=32 | 1.62 | — | — | 19,960 samples/s |
| Global | gRPC (`triton_server:8001`) | Batch=64 | 2.22 | — | — | 29,501 samples/s |
| Personalized | gRPC (`triton_server:8001`) | Single | 0.92 | 0.96 | 1.01 | 948 infer/s |
| Personalized | gRPC (`triton_server:8001`) | Batch=32 | 1.25 | — | — | 25,487 samples/s |
| Personalized | gRPC (`triton_server:8001`) | Batch=64 | 1.65 | — | — | 38,201 samples/s |

---

## HTTP vs gRPC Comparison

| Scenario | HTTP p50 (ms) | gRPC p50 (ms) | Speedup (HTTP/gRPC) | Winner |
|----------|------------:|------------:|-------------------:|--------|
| Global single | 0.92 | 1.02 | 0.90x | HTTP |
| Global batch=32 | 1.30 | 1.62 | 0.81x | HTTP |
| Global batch=64 | 1.72 | 2.22 | 0.78x | HTTP |
| Personal single | 0.84 | 0.92 | 0.92x | HTTP |
| Personal batch=32 | 1.24 | 1.25 | 1.00x | Tie |
| Personal batch=64 | 1.78 | 1.65 | 1.08x | gRPC |

HTTP wins for most scenarios. gRPC only wins on the largest personalized batch (b=64). The payloads are tiny (768 floats = 3KB), so HTTP's simpler protocol has less overhead than gRPC's protobuf + HTTP/2 framing.

---

## Production Pipeline Summary

| Component | Variant | Metric | Value |
|-----------|---------|--------|------:|
| ViT-L/14 | Compiled, bs=128 | Throughput (FPS) | 927.5 |
| | | Latency p50 (ms) | 137.94 |
| | | GPU mem (MB) | 1,947 |
| Global MLP (CPU) | Dynamic INT8 | Batch FPS (b=32) | 111,556 |
| | | Single FPS | 11,038 |
| | | Model size (MB) | 0.47 |
| Personalized MLP (CPU) | Graph-optimized | Batch FPS (b=32) | 91,549 |
| | | Single FPS | 16,886 |
| | | Model size (MB) | 1.90 |
| Global MLP (Triton HTTP) | 2 inst + dyn batch | Single (infer/s) | 597 |
| | | Batch=32 (samp/s) | 36,752 |
| | | Batch=64 (samp/s) | 62,847 |
| Personalized MLP (Triton HTTP) | 2 inst + dyn batch | Single (infer/s) | 593 |
| | | Batch=32 (samp/s) | 38,202 |
| | | Batch=64 (samp/s) | 54,729 |
| Global MLP (Triton gRPC) | 2 inst + dyn batch | Single (infer/s) | 946 |
| | | Batch=32 (samp/s) | 19,960 |
| | | Batch=64 (samp/s) | 29,501 |
| Personalized MLP (Triton gRPC) | 2 inst + dyn batch | Single (infer/s) | 948 |
| | | Batch=32 (samp/s) | 25,487 |
| | | Batch=64 (samp/s) | 38,201 |

---

## Production Workload Estimates

| Workload | Images | Time |
|----------|-------:|-----:|
| Onboarding (1 user, 5K photos) | 5,000 | 5.5s |
| └ ViT encoding | | 5.4s |
| └ Global MLP scoring (Triton) | | 0.08s |
| Daily new photos (1 user, 5 photos) | 5 | 8.4ms |
| Weekly re-score (100 users × 5K) | 500,000 | 9.1s |
| Weekly re-score (1000 users × 5K) | 5,000,000 | 91.4s |

---

## Triton Production Configuration

```
# flickr_global
name: "flickr_global"
platform: "onnxruntime_onnx"
max_batch_size: 128
instance_group: [{ count: 2, kind: KIND_GPU, gpus: [0] }]
dynamic_batching: { preferred_batch_size: [4, 8, 16, 32, 64, 128], max_queue_delay_microseconds: 50 }

# flickr_personalized
name: "flickr_personalized"
platform: "onnxruntime_onnx"
max_batch_size: 128
instance_group: [{ count: 2, kind: KIND_GPU, gpus: [0] }]
dynamic_batching: { preferred_batch_size: [4, 8, 16, 32, 64, 128], max_queue_delay_microseconds: 50 }
```

Key design decisions:
- 2 instances: +38% throughput, -49% queue delay at concurrency=8
- Dynamic batching ON: groups concurrent single-image requests from multiple users
- preferred_batch_size [4–128]: future-proofed for traffic growth
- max_queue_delay 50 µs: low enough for sparse arrivals
- Always-on server: users across time zones
- HTTP preferred over gRPC: faster for this payload size (768 floats = 3KB)
