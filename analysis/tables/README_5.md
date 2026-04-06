# Notebook 5: PyTorch Inference Benchmarks (GPU)

Hardware: NVIDIA A100 80GB + AMD EPYC 7763 CPU (Chameleon Cloud `compute_gigaio` at CHI@UC)

## Model Sizes

| Model | Size on Disk |
|-------|-------------|
| CLIP ViT-L/14 | 932.77 MB |
| Global MLP (768→512→128→32→1) | 1.86 MB |
| Personalized MLP (768+16→512→128→32→1) | 1.90 MB |

## Part 1: CLIP ViT-L/14 Image Encoder (GPU)

### Single Image Latency

| Metric | Eager | Compiled |
|--------|------:|--------:|
| Median latency (ms) | 10.29 | 5.39 |
| p95 latency (ms) | 10.55 | 6.44 |
| p99 latency (ms) | 13.80 | 12.64 |
| Throughput (FPS) | 95.92 | 175.03 |
| GPU util avg (%) | 21.0 | 27.5 |

### Batch Throughput

| Batch Size | Eager FPS | Compiled FPS | Eager p50 (ms) | Compiled p50 (ms) | Eager GPU mem (MB) | Compiled GPU mem (MB) |
|-----------|--------:|----------:|-------------:|----------------:|------------------:|--------------------:|
| 32 | 597.9 | 831.7 | 53.52 | 38.39 | 2,804 | 2,754 |
| 64 | 619.1 | 847.1 | 103.44 | 75.44 | 3,085 | 2,984 |
| 128 | 627.4 | 877.0 | 203.91 | 145.89 | 3,644 | 3,444 |
| 256 | 632.8 | 876.8 | 404.50 | 291.86 | 4,760 | 4,360 |
| 512 | 633.5 | 839.3 | 808.14 | 609.97 | 6,992 | 6,193 |
| 1024 | 604.1 | 873.7 | 1,694.97 | 1,172.14 | 11,457 | 9,857 |

Best: Compiled, batch_size=128 — peak throughput (877 FPS) at lowest memory (3.4 GB).

## Part 2: Global MLP Head (GPU)

### Single Sample Latency

| Metric | Eager | Compiled |
|--------|------:|--------:|
| Median latency (ms) | 0.19 | 0.42 |
| p95 latency (ms) | 0.34 | 0.62 |
| p99 latency (ms) | 0.35 | 0.85 |
| Throughput (FPS) | 4,318 | 2,259 |
| GPU mem peak (MB) | 2,523 | 2,526 |

### Batch Throughput

| Batch Size | Eager FPS | Compiled FPS | Eager p50 (ms) | Compiled p50 (ms) | Eager GPU mem (MB) | Compiled GPU mem (MB) |
|-----------|--------:|----------:|-------------:|----------------:|------------------:|--------------------:|
| 32 | 157,326 | 133,272 | 0.1917 | 0.2165 | 2,527 | 2,526 |
| 64 | 328,554 | 171,347 | 0.1889 | 0.3575 | 2,528 | 2,527 |
| 128 | 279,646 | 343,479 | 0.4507 | 0.3536 | 2,528 | 2,527 |
| 256 | 562,540 | 506,745 | 0.4443 | 0.4854 | 2,528 | 2,527 |
| 512 | 1,156,912 | 1,055,887 | 0.4282 | 0.4731 | 2,529 | 2,528 |
| 1024 | 2,327,341 | 2,102,861 | 0.4344 | 0.4802 | 2,531 | 2,529 |

Note: Eager mode is faster than compiled for the Global MLP. The model is so small that `torch.compile` overhead exceeds any fusion benefit.

## Part 3: Personalized MLP Head (GPU)

168 known users, user embedding dim = 16.

### Single Sample Latency

| Metric | Eager | Compiled |
|--------|------:|--------:|
| Median latency (ms) | 0.25 | 0.36 |
| p95 latency (ms) | 0.43 | 0.51 |
| p99 latency (ms) | 0.50 | 0.65 |
| Throughput (FPS) | 3,301 | 2,633 |
| GPU mem peak (MB) | 1,589 | 1,590 |

### Batch Throughput

| Batch Size | Eager FPS | Compiled FPS | Eager p50 (ms) | Compiled p50 (ms) | Eager GPU mem (MB) | Compiled GPU mem (MB) |
|-----------|--------:|----------:|-------------:|----------------:|------------------:|--------------------:|
| 32 | 124,313 | 130,374 | 0.2455 | 0.2345 | 1,591 | 1,590 |
| 64 | 254,760 | 164,565 | 0.2420 | 0.3874 | 1,591 | 1,590 |
| 128 | 502,039 | 245,041 | 0.2487 | 0.5068 | 1,591 | 1,590 |
| 256 | 1,020,144 | 497,831 | 0.2450 | 0.5075 | 1,592 | 1,591 |
| 512 | 2,087,083 | 983,451 | 0.2408 | 0.4998 | 1,594 | 1,592 |
| 1024 | 4,137,415 | 1,951,070 | 0.2401 | 0.5066 | 1,598 | 1,596 |

Note: Eager mode is ~2x faster than compiled for the Personalized MLP at larger batch sizes. Same reason as Global MLP — the model is too small for `torch.compile` to help.

### Sample Predictions (3 users, same batch of 32 images)

| User | Mean Score | Std | First 3 Scores |
|------|--------:|----:|----------------|
| 0 | 0.398 | 0.120 | 0.587, 0.264, 0.397 |
| 84 | 0.523 | 0.171 | 0.796, 0.288, 0.633 |
| 167 | 0.480 | 0.114 | 0.478, 0.391, 0.511 |

## Part 4: End-to-End Pipeline (GPU)

Image → ViT (GPU) → normalize → MLP (GPU) → score. All eager mode (uncompiled).

| Scenario | Median Latency (ms) | p95 Latency (ms) | Throughput (FPS) |
|----------|-------------------:|----------------:|----------------:|
| E2E single image (Global) | 10.59 | 10.90 | 92.74 |
| E2E single image (Personalized) | 10.82 | 10.97 | 90.65 |
| E2E batch=32 (Global) | — | — | 592.38 |
| E2E batch=32 (Personalized) | — | — | 591.15 |

### Latency Breakdown (single image)

| Component | Latency (ms) | % of E2E (Global) |
|-----------|------------:|------------------:|
| ViT encode | 10.29 | 97.2% |
| Global MLP forward | 0.19 | 1.8% |
| Personal MLP forward | 0.25 | 2.3% |
| **E2E total (Global)** | **10.59** | 100% |
| **E2E total (Personal)** | **10.82** | 100% |

The ViT encoder dominates the pipeline cost (~97%). MLP optimizations improve MLP latency but have negligible impact on total pipeline time.

## Part 5: Quality Metrics

### Global MLP (test split: 4,049 images)

| Metric | Value |
|--------|------:|
| MAE | 0.0729 |
| RMSE | 0.0936 |
| PLCC | 0.7929 |
| SRCC | 0.7729 |
| Binary accuracy (threshold=0.5) | 0.8271 |
| AUC-ROC | 0.8730 |

### Personalized MLP (test split: 15,453 rows, 168 users)

| Metric | Value |
|--------|------:|
| MAE (pooled) | 0.1421 |
| RMSE (pooled) | 0.1842 |
| PLCC (pooled) | 0.7378 |
| SRCC (pooled) | 0.7297 |
| Binary accuracy (pooled, threshold=0.5) | 0.7582 |
| AUC-ROC (pooled) | 0.8725 |
| Per-user SRCC (avg ± std) | 0.5919 ± 0.2402 |
| Per-user MAE (avg ± std) | 0.1721 ± 0.0470 |

### Personalization Gain (personalized vs global, same images)

| Metric | Global | Personalized | Improvement |
|--------|------:|------------:|------------|
| MAE | 0.2013 | 0.1421 | +0.0592 better |
| SRCC | 0.5979 | 0.7297 | +0.1317 better |
