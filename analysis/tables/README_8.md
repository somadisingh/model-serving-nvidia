# Notebook 8: GPU Execution Providers Benchmark

Hardware: NVIDIA A100 80GB + AMD EPYC 7763 CPU (Chameleon Cloud).

---

## Part 1: CLIP ViT-L/14 GPU

### Batch Throughput

| Batch Size | Eager p50 (ms) | Eager FPS | Compiled p50 (ms) | Compiled FPS |
|-----------|-------------:|--------:|----------------:|----------:|
| 1 | 9.38 | 106.5 | 4.99 | 200.2 |
| 8 | 14.60 | 549.3 | 11.33 | 709.6 |
| 32 | 52.61 | 605.3 | 38.58 | 829.6 |
| 64 | 101.88 | 627.4 | 75.62 | 846.0 |
| 128 | 201.45 | 635.6 | 147.01 | 870.1 |
| 256 | 402.80 | 635.7 | 292.43 | **875.8** |
| 512 | 808.65 | 632.9 | 612.64 | 835.8 |

---

## Part 2: End-to-End Pipeline (GPU ViT + CPU MLP)

| Scenario | Median (ms) | p95 (ms) | Throughput (FPS) |
|----------|----------:|--------:|----------------:|
| E2E single image (Global) | 9.88 | 10.15 | 100.77 |
| E2E single image (Personalized) | 9.85 | 9.93 | 100.94 |
| E2E batch=32 (Global) | — | — | 581.90 |
| E2E batch=32 (Personalized) | — | — | 583.21 |

---

## Part 3: MLP ONNX Execution Providers

All benchmarks use pre-computed CLIP embeddings (768-dim float32). Batch size = 32 for batch throughput.

### Global MLP — All Execution Providers

| Execution Provider | Single p50 (ms) | Single p95 (ms) | Single p99 (ms) | Single FPS | Batch=32 FPS | GPU util avg | GPU mem (MB) |
|-------------------|---------------:|---------------:|---------------:|---------:|----------:|----------:|----------:|
| CPUExecutionProvider | 0.04 | 0.05 | 0.10 | 25,976 | 184,345 | — | — |
| CUDAExecutionProvider | 0.09 | 0.11 | 0.11 | 10,307 | 262,575 | 0.0% | 57,440 |
| TensorrtExecutionProvider | 0.10 | 0.12 | 0.13 | 9,383 | 224,452 | 5.1% | 57,625 |
| OpenVINOExecutionProvider | 0.43 | 0.46 | 0.48 | 2,325 | 60,356 | — | — |

### Personalized MLP — All Execution Providers

| Execution Provider | Single p50 (ms) | Single p95 (ms) | Single p99 (ms) | Single FPS | Batch=32 FPS | GPU util avg | GPU mem (MB) |
|-------------------|---------------:|---------------:|---------------:|---------:|----------:|----------:|----------:|
| CPUExecutionProvider | 0.03 | 0.04 | 0.06 | 31,669 | 176,639 | — | — |
| CUDAExecutionProvider | 0.18 | 0.23 | 0.25 | 5,220 | 154,259 | 0.0% | 57,448 |
| TensorrtExecutionProvider | 0.11 | 0.13 | 0.15 | 8,527 | 205,748 | 8.0% | 57,788 |
| OpenVINOExecutionProvider | 0.09 | 0.11 | 0.11 | 10,966 | 201,959 | — | — |

---

## Quality Metrics — TensorRT EP (Global MLP, N=4,049)

| Metric | TensorRT EP | FP32 Baseline (NB6) | Delta |
|--------|----------:|------------------:|------:|
| MAE | 0.0729 | 0.0729 | 0.0000 |
| RMSE | 0.0936 | 0.0936 | 0.0000 |
| PLCC | 0.7929 | 0.7929 | 0.0000 |
| SRCC | 0.7729 | 0.7729 | 0.0000 |
| Binary accuracy | 0.8271 | 0.8271 | 0.0000 |
| AUC-ROC | 0.8730 | 0.8730 | 0.0000 |

No precision loss — TRT on A100 uses TF32 by default (not FP16), which preserves FP32 accuracy.

## Quality Metrics — TensorRT EP (Personalized MLP, 162 users)

| Metric | TensorRT EP | FP32 Baseline (NB6) | Delta |
|--------|----------:|------------------:|------:|
| Mean per-user SRCC | 0.5918 | 0.5919 | -0.0001 |
| Mean per-user MAE | 0.1721 | 0.1721 | 0.0000 |

Negligible difference — TF32 precision is preserved.

---

## Summary: Best EP per Scenario

| Scenario | Best EP | Reason |
|----------|---------|--------|
| Single-sample latency | CPU | Lowest overhead for tiny model (0.03–0.04 ms) |
| Batch throughput (Global) | CUDA | 262K samples/s vs 224K (TRT) and 184K (CPU) |
| Batch throughput (Personalized) | TRT | 206K samples/s vs 177K (CPU) and 154K (CUDA) |
| Quality preservation | TRT / CUDA | Both identical to FP32 baseline |
| Edge/CPU-only deployment | OpenVINO | 2,325 FPS single (Global), but broken for Personalized |

---

## OpenVINO Execution Provider — Details

Ran in a separate container (`jupyter-onnx-openvino`) since ONNX Runtime cannot load CUDA/TRT and OpenVINO EPs simultaneously.

### Global MLP — OpenVINO (correct)

| Metric | Value |
|--------|------:|
| Execution provider | OpenVINOExecutionProvider, CPUExecutionProvider |
| Device | CPU-OPENVINO_CPU |
| Sample scores (first 5) | 0.62, 0.50, 0.57, 0.57, 0.54 |
| Mean predicted score | 0.54 (std: 0.09) |
| Single latency (p50) | 0.43 ms |
| Single latency (p95) | 0.46 ms |
| Single latency (p99) | 0.48 ms |
| Single throughput | 2,325 FPS |
| Batch=32 throughput | 60,356 FPS |

Scores match FP32 baseline — OpenVINO works correctly for the Global MLP.

### Personalized MLP — OpenVINO (BROKEN)

| Metric | Value |
|--------|------:|
| Execution provider | OpenVINOExecutionProvider, CPUExecutionProvider |
| Device | CPU-OPENVINO_CPU |
| Sample scores (first 5) | 0.35, 0.00, 0.00, 0.00, 0.00 |
| Mean predicted score | 0.05 (std: 0.16) |
| Single latency (p50) | 0.09 ms |
| Single latency (p95) | 0.11 ms |
| Single latency (p99) | 0.11 ms |
| Single throughput | 10,966 FPS |
| Batch=32 throughput | 201,959 FPS |

Scores are incorrect — most outputs are 0.00. The OpenVINO EP does not correctly handle the `nn.Embedding` lookup (user_idx → 16-dim vector) in the personalized model. The performance numbers are fast but meaningless since the model produces wrong results. OpenVINO is not suitable for the Personalized MLP.
