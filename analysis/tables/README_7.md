# Notebook 7: ONNX Optimization Benchmarks (CPU)

Hardware: AMD EPYC 7763 CPU, NVIDIA A100 80GB GPU (for CLIP encoding only). All ONNX sessions use `CPUExecutionProvider`. Quantization via Intel Neural Compressor 2.6.

**Important notes:**
- Global MLP conservative static quantization: `quantization.fit()` returned `None` — no quantizable ops found. Saved as FP32 fallback (identical to graph-optimized).
- Personalized MLP conservative static quantization: same — `None` returned, saved as FP32 fallback.
- Personalized MLP aggressive static quantization: technically quantized but size barely changed (1.89 MB vs 1.90 MB), suggesting minimal INT8 conversion.

---

## Global MLP — All Variants

### Performance Summary

| Variant | Size (MB) | Single p50 (ms) | Single p95 (ms) | Single p99 (ms) | Single FPS | Batch=32 FPS |
|---------|--------:|---------------:|---------------:|---------------:|---------:|----------:|
| FP32 ONNX (no opts, NB6 baseline) | 1.86 | 0.03 | 0.04 | 0.05 | 32,610 | 322,980 |
| Graph-optimized (ORT_ENABLE_EXTENDED) | 1.86 | 0.02 | 0.04 | 0.07 | 35,569 | 220,535 |
| Dynamic INT8 (INC) | **0.47** | 0.03 | 0.04 | 0.05 | 35,530 | 394,572 |
| Static aggressive (quant_level=1) | 1.86 | 0.02 | 0.03 | 0.06 | 38,445 | 412,039 |
| Static conservative (quant_level=0) | 1.86 | 0.02 | 0.03 | 0.11 | 36,415 | 414,841 |

Note: Static aggressive and conservative both fell back to FP32 with graph optimizations (no quantizable ops found). Their sizes remain 1.86 MB.

### Quality Metrics (N=4,049 test images)

| Variant | MAE | RMSE | PLCC | SRCC | Acc (t=0.5) | AUC-ROC |
|---------|----:|-----:|-----:|-----:|----------:|-------:|
| Graph-optimized | 0.0729 | 0.0936 | 0.7929 | 0.7729 | 0.8271 | 0.8730 |
| Dynamic INT8 | 0.0729 | 0.0935 | 0.7931 | 0.7731 | 0.8269 | 0.8730 |
| Static aggressive (FP32 fallback) | 0.0729 | 0.0936 | 0.7929 | 0.7729 | 0.8271 | 0.8730 |
| Static conservative (FP32 fallback) | 0.0729 | 0.0936 | 0.7929 | 0.7729 | 0.8271 | 0.8730 |

All variants preserve identical quality. Dynamic INT8 is the only truly quantized model.

---

## Personalized MLP — All Variants

### Performance Summary

| Variant | Size (MB) | Single p50 (ms) | Single p95 (ms) | Single p99 (ms) | Single FPS | Batch=32 FPS |
|---------|--------:|---------------:|---------------:|---------------:|---------:|----------:|
| FP32 ONNX (no opts, NB6 baseline) | 1.90 | 0.04 | 0.06 | 0.08 | 24,530 | 93,369 |
| Graph-optimized (ORT_ENABLE_EXTENDED) | 1.90 | 0.03 | 0.04 | 0.06 | 32,406 | 292,796 |
| Dynamic INT8 (INC) | **0.49** | 0.03 | 0.04 | 0.05 | 34,382 | 359,525 |
| Static aggressive (quant_level=1) | 1.89 | 0.03 | 0.05 | 0.39 | 24,482 | 302,497 |
| Static conservative (quant_level=0) | 1.90 | 0.03 | 0.04 | 0.06 | 31,833 | 304,777 |

Note: Static conservative fell back to FP32 (no quantizable ops). Static aggressive technically ran but size barely changed (1.89 MB), indicating minimal quantization.

### Quality Metrics (N=15,453 rows, 162 users)

| Variant | Mean per-user SRCC | Mean per-user MAE |
|---------|------------------:|----------------:|
| Graph-optimized | 0.5920 | 0.1721 |
| Dynamic INT8 | 0.5740 | 0.1740 |
| Static aggressive | 0.5877 | 0.1793 |
| Static conservative (FP32 fallback) | 0.5920 | 0.1721 |

---

## Summary: Best Variant per Model

| Model | Best Variant | Reason |
|-------|-------------|--------|
| Global MLP | Dynamic INT8 | 75% smaller (0.47 MB), identical quality, competitive throughput |
| Personalized MLP | Graph-optimized | Simpler (no INC dependency), identical quality to conservative static, 3x batch throughput vs baseline |

The static quantization approaches failed to find quantizable ops for both models on this hardware/INC version combination. Dynamic INT8 is the only variant that achieves genuine compression.
