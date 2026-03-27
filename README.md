# Model Optimizations for Aesthetic Score Prediction

This lab benchmarks and optimizes a two-stage image aesthetic scoring pipeline on NVIDIA GPU hardware (A100/A30) and AMD EPYC 7763 CPU.

## Pipeline

```
Image (3840px wide) → Resize/CenterCrop to 224×224
    → CLIP ViT-L/14 (304M params, frozen, GPU) → 768-dim embedding
    → MLP (928K params, CPU) → scalar aesthetic score (0–10)
```

- **CLIP ViT-L/14**: A frozen Vision Transformer that generates 768-dimensional image embeddings. Benchmarked on latency and throughput across batch sizes on both CPU and GPU (eager vs compiled mode).
- **MLP**: A lightweight feed-forward network (768→1024→128→64→16→1) that takes the CLIP embedding and outputs a continuous aesthetic score. Benchmarked on latency, throughput, and sample prediction quality.

## Dataset

Images sourced from Pixabay, organized into `test/` and `validation/` splits with flat numbered filenames (`1.jpeg`, `2.jpeg`, ...). Metadata and tags are stored in accompanying CSV files (`uhd-iqa-metadata.csv`, `uhd-iqa-tags.csv`). Images were originally larger than UHD-1 (3840×2160) and have been downscaled to 3840px width maintaining aspect ratio.

## What this lab covers

1. **Data preparation**: Downloading the dataset into a Docker volume for use by Jupyter containers.
2. **PyTorch baseline**: Measuring ViT encoder latency/throughput (eager vs compiled, CPU and GPU) and MLP inference latency/throughput on CPU. End-to-end pipeline timing on both CPU and GPU.
3. **ONNX conversion**: Exporting the MLP head to ONNX format.
4. **ONNX optimization**: Graph optimizations, dynamic and static quantization on the MLP model.
5. **Execution providers**: Testing the MLP ONNX model with CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider, and OpenVINOExecutionProvider.

## Hardware

This lab uses NVIDIA GPU nodes on Chameleon Cloud:

- `compute_liqid` nodes at CHI@TACC — NVIDIA A100 40GB + AMD EPYC 7763 CPU
- `compute_gigaio` nodes at CHI@UC — NVIDIA A100 80GB + AMD EPYC 7763 CPU
