# Model Optimizations for Aesthetic Score Prediction

This lab benchmarks and optimizes a two-stage image aesthetic scoring pipeline on NVIDIA GPU hardware (A100/A30) and AMD EPYC 7763 CPU.

## Pipeline

```
Image (3840px wide) → Resize/CenterCrop to 224×224
    → CLIP ViT-L/14 (304M params, frozen, GPU) → 768-dim embedding
    → MLP (928K params, CPU) → scalar aesthetic score (0–10)
```

- **CLIP ViT-L/14**: A frozen Vision Transformer that generates 768-dimensional image embeddings on GPU. Benchmarked on latency and throughput only (no accuracy — it's a feature extractor, not a predictor).
- **MLP**: A lightweight feed-forward network (768→1024→128→64→16→1) that takes the CLIP embedding and outputs a continuous aesthetic score. Benchmarked on latency, throughput, and prediction quality (MAE/MSE against ground truth scores).

## Dataset

Images sourced from Pixabay, organized into `train/`, `test/`, `validation/` splits with flat numbered filenames (`1.jpeg`, `2.jpeg`, ...). Labels (aesthetic scores) and metadata are stored in accompanying CSV files (`metadata.csv`, `tags.csv`). Images were originally larger than UHD-1 (3840×2160) and have been downscaled to 3840px width maintaining aspect ratio.

## What this lab covers

1. **Data preparation**: Downloading the dataset into a Docker volume for use by Jupyter containers.
2. **PyTorch baseline**: Measuring embedding generation latency/throughput (CLIP on GPU) and inference latency/throughput/MAE/MSE (MLP on CPU).
3. **ONNX conversion**: Exporting both CLIP ViT and MLP to ONNX format separately.
4. **ONNX optimization**: Graph optimizations, dynamic and static quantization on both models.
5. **Execution providers**: Testing CLIP with CUDAExecutionProvider and TensorrtExecutionProvider on GPU; testing MLP with CPUExecutionProvider and OpenVINOExecutionProvider on CPU.

## Hardware

This lab uses NVIDIA GPU nodes on Chameleon Cloud:

- `compute_liqid` nodes at CHI@TACC — NVIDIA A100 40GB + AMD EPYC 7763 CPU
- `compute_gigaio` nodes at CHI@UC — NVIDIA A100 80GB + AMD EPYC 7763 CPU
