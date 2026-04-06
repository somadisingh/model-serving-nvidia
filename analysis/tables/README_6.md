# Notebook 6: ONNX Inference Benchmarks (CPU)

Hardware: AMD EPYC 7763 CPU, NVIDIA A100 80GB GPU (for CLIP encoding only). ONNX sessions use `CPUExecutionProvider`.

Graph optimizations are explicitly disabled (`ORT_DISABLE_ALL`) to establish a clean FP32 baseline before optimization in notebook 7.

## Model Sizes

| Model | Format | Size on Disk |
|-------|--------|-------------|
| Global MLP (FP32 ONNX) | `flickr_global.onnx` | 1.86 MB |
| Personalized MLP (FP32 ONNX) | `flickr_personalized.onnx` | 1.90 MB |

## Global MLP — FP32 ONNX Baseline (CPU, no graph optimizations)

### Performance

| Metric | Value |
|--------|------:|
| Single sample latency (p50) | 0.03 ms |
| Single sample latency (p95) | 0.04 ms |
| Single sample latency (p99) | 0.05 ms |
| Single sample throughput | 32,610 FPS |
| Batch throughput (b=32) | 322,980 FPS |

### Sample Predictions

| Image | Score |
|-------|------:|
| Image 1 | 0.62 |
| Image 2 | 0.50 |
| Image 3 | 0.57 |
| Image 4 | 0.57 |
| Image 5 | 0.54 |

Batch mean: 0.54, std: 0.09

### Quality Metrics (test split: 4,049 images)

| Metric | Value |
|--------|------:|
| N | 4,049 |
| MAE | 0.0729 |
| RMSE | 0.0936 |
| PLCC | 0.7929 |
| SRCC | 0.7729 |
| Binary accuracy (threshold=0.5) | 0.8271 |
| AUC-ROC | 0.8730 |

## Personalized MLP — FP32 ONNX Baseline (CPU, no graph optimizations)

168 known users, user embedding dim = 16. Inputs: `embedding` [batch_size, 768] (float32) + `user_idx` [batch_size] (int64).

### Performance

| Metric | Value |
|--------|------:|
| Single sample latency (p50) | 0.04 ms |
| Single sample latency (p95) | 0.06 ms |
| Single sample latency (p99) | 0.08 ms |
| Single sample throughput | 24,530 FPS |
| Batch throughput (b=32) | 93,369 FPS |

### Sample Predictions (user_idx=0)

| Image | Score |
|-------|------:|
| Image 1 | 0.59 |
| Image 2 | 0.26 |
| Image 3 | 0.40 |
| Image 4 | 0.42 |
| Image 5 | 0.42 |

Batch mean: 0.40, std: 0.12

### Quality Metrics (test split: 15,453 rows, 162 users evaluated)

| Metric | Value |
|--------|------:|
| Mean per-user SRCC | 0.5919 |
| Mean per-user MAE | 0.1721 |

## Summary Comparison

| Model | Size (MB) | Single FPS | Batch=32 FPS | Single p50 (ms) |
|-------|--------:|----------:|-----------:|---------------:|
| Global MLP (FP32 ONNX) | 1.86 | 32,610 | 322,980 | 0.03 |
| Personalized MLP (FP32 ONNX) | 1.90 | 24,530 | 93,369 | 0.04 |

Both models are extremely fast on CPU even without graph optimizations. The personalized model is ~3.5x slower at batch throughput due to the additional embedding lookup and concatenation operations.
