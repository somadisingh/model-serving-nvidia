# Model Optimizations for Aesthetic Score Prediction

This lab benchmarks and optimizes a two-stage image aesthetic scoring pipeline on NVIDIA GPU hardware (A100/A30) and AMD EPYC 7763 CPU. It can be accessed [here](https://trovi.chameleoncloud.org/dashboard/artifacts/c347ab71-1a5b-41cf-a2fd-0c34d30f1e1d/).

## Pipeline

```
Image (~500×500 Flickr JPG) → Resize/CenterCrop to 224×224
    → CLIP ViT-L/14 (304M params, frozen, GPU) → 768-dim embedding
    → Global MLP (~430K params) → scalar aesthetic score (0–1)
    → Personalized MLP (~430K params + user embeddings) → per-user aesthetic score (0–1)
```

- **CLIP ViT-L/14**: A frozen Vision Transformer that generates 768-dimensional image embeddings. Benchmarked on latency and throughput across batch sizes on both CPU and GPU (eager vs compiled mode).
- **Global MLP**: A lightweight feed-forward network (768→512→128→32→1 with sigmoid) that predicts a universal aesthetic score.
- **Personalized MLP**: Takes a 768-dim CLIP embedding + 64-dim learned user embedding (832 total) and predicts a per-user aesthetic score.

## Dataset

Flickr-AES (ICCV 2017 Aesthetics Dataset) — ~40K Flickr images with crowd-sourced aesthetic ratings, hosted on Google Drive. Split manifests (`flickr_global_manifest.csv`, `flickr_personalized_manifest.csv`) define train/val/test/production splits. Data is downloaded via `gdown` into a Docker volume.

## What this lab covers

1. **Data preparation**: Downloading the Flickr-AES dataset from Google Drive into a Docker volume using `gdown`.
2. **PyTorch baseline**: Measuring ViT encoder latency/throughput (eager vs compiled, CPU and GPU) and MLP inference latency/throughput on CPU. End-to-end pipeline timing on both CPU and GPU.
3. **ONNX conversion**: Exporting the MLP heads to ONNX format.
4. **ONNX optimization**: Graph optimizations, dynamic and static quantization on the MLP models.
5. **Execution providers**: Testing the MLP ONNX models with CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider, and OpenVINOExecutionProvider.
6. **FastAPI serving**: Deploying the optimized MLP heads as a REST API (FastAPI + ONNX Runtime GPU) with four endpoints (global/personalized × single/batch). Benchmarked for latency and throughput under sequential and concurrent load.
7. **Triton Inference Server**: Serving both models via NVIDIA Triton (ONNX backend). Benchmarked with the Python `tritonclient`, `perf_analyzer` concurrency/batch sweeps, multi-instance scaling, and dynamic batching.

## Hardware

This lab uses NVIDIA GPU nodes on Chameleon Cloud:

- `compute_liqid` nodes at CHI@TACC — NVIDIA A100 40GB + AMD EPYC 7763 CPU
- `compute_gigaio` nodes at CHI@UC — NVIDIA A100 80GB + AMD EPYC 7763 CPU
