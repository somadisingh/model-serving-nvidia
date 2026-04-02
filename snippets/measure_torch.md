
::: {.cell .markdown}

## Measure inference performance of PyTorch model on CPU 

First, we are going to measure the inference performance of an already-trained PyTorch model on CPU. Our full inference pipeline has two stages:

1. **CLIP ViT-L/14** (image encoder): Takes a raw image and produces a 768-dimensional embedding vector
2. **Aesthetic MLP head**: Takes the 768-dim embedding and produces an aesthetic quality score (0-1)

We will benchmark each stage independently, then measure the end-to-end pipeline. After completing this section, you should understand:

* how to measure the inference latency and throughput of a PyTorch model
* the relative cost of the image encoder (ViT) vs the downstream head (MLP)
* how to compare eager model execution vs a compiled model

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import time
import numpy as np
import pandas as pd
import clip
```
:::


::: {.cell .markdown}

## Resource monitoring

The `ResourceMonitor` class polls `nvidia-smi` (GPU utilization and memory) and `psutil` (CPU and RAM) in a background thread alongside each benchmark. The results tell you how much GPU, CPU, and RAM your workload actually needs — useful for right-sizing the instance.

:::

::: {.cell .code}
```python
import subprocess
import threading
import psutil


class ResourceMonitor:
    """Polls nvidia-smi (GPU) and psutil (CPU/RAM) in a background thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.gpu_util = []
        self.gpu_mem_used = []
        self.cpu_percent = []
        self.ram_used_gb = []
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True
                ).strip().split(",")
                self.gpu_util.append(float(out[0]))
                self.gpu_mem_used.append(float(out[1]))
            except Exception:
                pass  # nvidia-smi unavailable — GPU metrics skipped
            self.cpu_percent.append(psutil.cpu_percent(interval=None))
            self.ram_used_gb.append(psutil.virtual_memory().used / 1e9)
            time.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self.gpu_util.clear()
        self.gpu_mem_used.clear()
        self.cpu_percent.clear()
        self.ram_used_gb.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self, label=""):
        print(f"\nResource usage — {label}")
        if self.gpu_util:
            print(f"  GPU util:  avg={np.mean(self.gpu_util):5.1f}%  peak={max(self.gpu_util):5.1f}%")
            print(f"  GPU mem:   avg={np.mean(self.gpu_mem_used):6.0f} MB  peak={max(self.gpu_mem_used):6.0f} MB")
        print(f"  CPU util:  avg={np.mean(self.cpu_percent):5.1f}%  peak={max(self.cpu_percent):5.1f}%")
        print(f"  RAM used:  avg={np.mean(self.ram_used_gb):5.2f} GB  peak={max(self.ram_used_gb):5.2f} GB")


monitor = ResourceMonitor()
print("ResourceMonitor ready.")
```
:::


::: {.cell .markdown}

First, let's load our MLP head and the CLIP ViT-L/14 model (used to compute image embeddings). Note that for now, we will use the CPU for inference, not GPU.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_path = "models/flickr_global_best_inference_only.pth"  
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Load CLIP model for computing image embeddings
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
```
:::

::: {.cell .markdown}

and also prepare our test dataset, using CLIP's own preprocessing:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```
:::


::: {.cell .markdown}

---

## Part 1: CLIP ViT-L/14 Image Encoder (CPU)

The ViT-L/14 model is the computationally expensive part of the pipeline. It processes raw images (224×224) through a Vision Transformer to produce 768-dimensional embeddings. Let's measure its performance on CPU in **eager mode** first.

:::

::: {.cell .markdown}

#### ViT model size

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# CLIP downloads the ViT model to ~/.cache/clip/
clip_cache_dir = os.path.expanduser("~/.cache/clip")
vit_model_file = os.path.join(clip_cache_dir, "ViT-L-14.pt")
if os.path.exists(vit_model_file):
    vit_model_size = os.path.getsize(vit_model_file)
    print(f"ViT-L/14 Model Size on Disk: {vit_model_size / (1e6):.2f} MB")
else:
    print(f"ViT model file not found at {vit_model_file}")
    # Estimate from parameters
    vit_params = sum(p.numel() * p.element_size() for p in clip_model.visual.parameters())
    print(f"ViT-L/14 Visual Encoder Size (in memory): {vit_params / (1e6):.2f} MB")
```
:::

::: {.cell .markdown}

#### ViT single-image latency (eager mode)

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Get a single preprocessed image
single_image, _ = next(iter(test_loader))
single_image = single_image[:1].to(device)

# Warm-up
with torch.no_grad():
    clip_model.encode_image(single_image)

monitor.start()
vit_latencies_eager = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        clip_model.encode_image(single_image)
        latencies_i = time.time() - start_time
        vit_latencies_eager.append(latencies_i)
monitor.stop()

print("ViT-L/14 Single Image Latency (Eager, CPU):")
print(f"  Median: {np.percentile(vit_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(vit_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(vit_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(vit_latencies_eager):.2f} FPS")
monitor.summary("ViT-L/14 eager single image (CPU)")
```
:::

::: {.cell .markdown}

#### ViT batch throughput (eager mode)

We'll test with a batch of 32 images (matching our DataLoader batch size).

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_batches = 50
batch_images, _ = next(iter(test_loader))
batch_images = batch_images.to(device)

# Warm-up
with torch.no_grad():
    clip_model.encode_image(batch_images)

vit_batch_times_eager = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        clip_model.encode_image(batch_images)
        vit_batch_times_eager.append(time.time() - start_time)

vit_batch_fps_eager = (batch_images.shape[0] * num_batches) / np.sum(vit_batch_times_eager)
print(f"ViT-L/14 Batch Throughput (Eager, CPU, batch_size=32): {vit_batch_fps_eager:.2f} FPS")
```
:::


::: {.cell .markdown}

#### ViT compiled mode (CPU)

Now let's compile the ViT visual encoder into a graph and see if we get a speedup. Graph compilation can fuse operations and optimize memory access patterns.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Compile the visual encoder
clip_model.visual = torch.compile(clip_model.visual)

# Warm-up (first call triggers compilation, which is slow)
print("Compiling ViT visual encoder (this may take a moment)...")
with torch.no_grad():
    clip_model.encode_image(single_image)
print("Compilation complete.")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Single-image latency (compiled)
vit_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        clip_model.encode_image(single_image)
        vit_latencies_compiled.append(time.time() - start_time)

print("ViT-L/14 Single Image Latency (Compiled, CPU):")
print(f"  Median: {np.percentile(vit_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(vit_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(vit_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(vit_latencies_compiled):.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Batch throughput (compiled)
vit_batch_times_compiled = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        clip_model.encode_image(batch_images)
        vit_batch_times_compiled.append(time.time() - start_time)

vit_batch_fps_compiled = (batch_images.shape[0] * num_batches) / np.sum(vit_batch_times_compiled)
print(f"ViT-L/14 Batch Throughput (Compiled, CPU, batch_size=32): {vit_batch_fps_compiled:.2f} FPS")
```
:::

::: {.cell .markdown}

#### ViT CPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 60)
print("CLIP ViT-L/14 CPU Benchmark Summary")
print("=" * 60)
print(f"{'Metric':<45} {'Eager':>8} {'Compiled':>8}")
print("-" * 60)
print(f"{'Single image latency (median, ms)':<45} {np.percentile(vit_latencies_eager, 50)*1000:>8.2f} {np.percentile(vit_latencies_compiled, 50)*1000:>8.2f}")
print(f"{'Single image latency (p95, ms)':<45} {np.percentile(vit_latencies_eager, 95)*1000:>8.2f} {np.percentile(vit_latencies_compiled, 95)*1000:>8.2f}")
print(f"{'Single image throughput (FPS)':<45} {num_trials/np.sum(vit_latencies_eager):>8.2f} {num_trials/np.sum(vit_latencies_compiled):>8.2f}")
print(f"{'Batch throughput (FPS, batch_size=32)':<45} {vit_batch_fps_eager:>8.2f} {vit_batch_fps_compiled:>8.2f}")
```
:::


::: {.cell .markdown}

---

## Part 2: Aesthetic MLP Head (CPU)

Now we'll benchmark the lightweight MLP head that maps 768-dim CLIP embeddings to aesthetic scores. Since we'll compare eager vs compiled mode, we'll first reload the models fresh.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload CLIP (uncompiled) and MLP for clean benchmarking
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()
```
:::


::: {.cell .markdown}

#### MLP model size

Our `flickr_global_best_inference_only.pth` is a lightweight MLP head (768 → 512 → 128 → 32 → 1) that maps CLIP ViT-L/14 embeddings to aesthetic scores, so it is very small.
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
mlp_model_size = os.path.getsize(model_path) 
print(f"MLP Model Size on Disk: {mlp_model_size / (1e6):.2f} MB")
```
:::

::: {.cell .markdown}

#### Sample predictions

Let's verify the model produces reasonable aesthetic scores.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    embeddings = torch.from_numpy(normalized(image_features.cpu().numpy())).float().to(device)
    scores = model(embeddings).squeeze()
    mean_score = scores.mean().item()
    std_score = scores.std().item()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (0-1):")
for i in range(min(5, len(scores))):
    print(f"  Image {i+1}: {scores[i].item():.2f}")
print(f"\nBatch mean: {mean_score:.2f}, std: {std_score:.2f}")
```
:::

::: {.cell .markdown}

#### MLP inference latency (eager mode)

We pre-compute a CLIP embedding, then measure only the MLP forward pass.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Pre-compute a single CLIP embedding for benchmarking MLP latency
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    single_embedding = torch.from_numpy(normalized(sample_features.cpu().numpy())).float().to(device)

# Warm-up run 
with torch.no_grad():
    model(single_embedding)

mlp_latencies_eager = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(single_embedding)
        mlp_latencies_eager.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("MLP Single Sample Latency (Eager, CPU):")
print(f"  Median: {np.percentile(mlp_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(mlp_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(mlp_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(mlp_latencies_eager):.2f} FPS")
```
:::

::: {.cell .markdown}

#### MLP batch throughput (eager mode)

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_batches = 50

# Pre-compute a batch of CLIP embeddings for benchmarking MLP throughput
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    batch_embeddings = torch.from_numpy(normalized(batch_features.cpu().numpy())).float().to(device)

# Warm-up run 
with torch.no_grad():
    model(batch_embeddings)

mlp_batch_times_eager = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = model(batch_embeddings)
        mlp_batch_times_eager.append(time.time() - start_time)

mlp_batch_fps_eager = (batch_embeddings.shape[0] * num_batches) / np.sum(mlp_batch_times_eager)
print(f"MLP Batch Throughput (Eager, CPU, batch_size=32): {mlp_batch_fps_eager:.2f} FPS")
```
:::


::: {.cell .markdown}

#### MLP compiled mode (CPU)

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model = torch.compile(model)

# Warm-up (triggers compilation)
print("Compiling MLP model (this may take a moment)...")
with torch.no_grad():
    model(single_embedding)
print("Compilation complete.")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
mlp_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(single_embedding)
        mlp_latencies_compiled.append(time.time() - start_time)

print("MLP Single Sample Latency (Compiled, CPU):")
print(f"  Median: {np.percentile(mlp_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(mlp_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(mlp_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(mlp_latencies_compiled):.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
mlp_batch_times_compiled = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = model(batch_embeddings)
        mlp_batch_times_compiled.append(time.time() - start_time)

mlp_batch_fps_compiled = (batch_embeddings.shape[0] * num_batches) / np.sum(mlp_batch_times_compiled)
print(f"MLP Batch Throughput (Compiled, CPU, batch_size=32): {mlp_batch_fps_compiled:.2f} FPS")
```
:::


::: {.cell .markdown}

#### MLP CPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 60)
print("Aesthetic MLP Head CPU Benchmark Summary")
print("=" * 60)
print(f"MLP Model Size on Disk: {mlp_model_size / (1e6):.2f} MB")
print(f"Mean Predicted Score: {mean_score:.2f} (std: {std_score:.2f})")
print()
print(f"{'Metric':<45} {'Eager':>8} {'Compiled':>8}")
print("-" * 60)
print(f"{'Single sample latency (median, ms)':<45} {np.percentile(mlp_latencies_eager, 50)*1000:>8.2f} {np.percentile(mlp_latencies_compiled, 50)*1000:>8.2f}")
print(f"{'Single sample latency (p95, ms)':<45} {np.percentile(mlp_latencies_eager, 95)*1000:>8.2f} {np.percentile(mlp_latencies_compiled, 95)*1000:>8.2f}")
print(f"{'Single sample throughput (FPS)':<45} {num_trials/np.sum(mlp_latencies_eager):>8.2f} {num_trials/np.sum(mlp_latencies_compiled):>8.2f}")
print(f"{'Batch throughput (FPS, batch_size=32)':<45} {mlp_batch_fps_eager:>8.2f} {mlp_batch_fps_compiled:>8.2f}")
```
:::

::: {.cell .markdown}

---

## Part 3: Personalized MLP Head (CPU)

The personalized model takes both a 768-dim CLIP embedding and a user index as input. It has an `nn.Embedding` table that maps each user index to a 64-dim learned vector, concatenates it with the CLIP embedding (832-dim total), and passes through the same MLP architecture. This lets the model learn per-user aesthetic preferences.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload CLIP (uncompiled) for clean benchmarking
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Load personalized model
personal_model_path = "models/flickr_personalized_best_inference_only.pth"
personal_model = torch.load(personal_model_path, map_location=device, weights_only=False)
personal_model.eval()

# Get valid user indices from the personalized manifest
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
personal_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
seen_workers = sorted(personal_manifest.loc[personal_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique())
user2idx = {u: i for i, u in enumerate(seen_workers)}
num_users = len(user2idx)
print(f"Personalized model: {num_users} known users, embedding table shape: {personal_model.user_embedding.weight.shape}")
```
:::

::: {.cell .markdown}

#### Personalized MLP model size

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
personal_mlp_model_size = os.path.getsize(personal_model_path)
print(f"Personalized MLP Model Size on Disk: {personal_mlp_model_size / (1e6):.2f} MB")
print(f"Global MLP Model Size on Disk:       {mlp_model_size / (1e6):.2f} MB")
```
:::


::: {.cell .markdown}

#### Sample predictions

Let's verify the personalized model produces reasonable scores for a few different users on the same images.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    embeddings = torch.from_numpy(normalized(image_features.cpu().numpy())).float().to(device)

    # Pick 3 different users and score the same batch
    sample_user_ids = [0, num_users // 2, num_users - 1]
    for uid in sample_user_ids:
        user_idx = torch.full((embeddings.shape[0],), uid, dtype=torch.long, device=device)
        scores = personal_model(embeddings, user_idx).squeeze()
        print(f"User {uid}: mean={scores.mean().item():.3f}, std={scores.std().item():.3f}, first 3: {[f'{s:.3f}' for s in scores[:3].tolist()]}")
```
:::


::: {.cell .markdown}

#### Personalized MLP inference latency (eager mode)

We pre-compute a CLIP embedding, then measure only the personalized MLP forward pass (embedding + user index → score).

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Pre-compute a single CLIP embedding for benchmarking
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    p_single_embedding = torch.from_numpy(normalized(sample_features.cpu().numpy())).float().to(device)
    p_single_user_idx = torch.tensor([0], dtype=torch.long, device=device)

# Warm-up run
with torch.no_grad():
    personal_model(p_single_embedding, p_single_user_idx)

personal_mlp_latencies_eager = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = personal_model(p_single_embedding, p_single_user_idx)
        personal_mlp_latencies_eager.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Personalized MLP Single Sample Latency (Eager, CPU):")
print(f"  Median: {np.percentile(personal_mlp_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(personal_mlp_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(personal_mlp_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(personal_mlp_latencies_eager):.2f} FPS")
```
:::


::: {.cell .markdown}

#### Personalized MLP batch throughput (eager mode)

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_batches = 50

# Pre-compute batch of CLIP embeddings
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    p_batch_embeddings = torch.from_numpy(normalized(batch_features.cpu().numpy())).float().to(device)
    p_batch_user_idx = torch.zeros(p_batch_embeddings.shape[0], dtype=torch.long, device=device)

# Warm-up run
with torch.no_grad():
    personal_model(p_batch_embeddings, p_batch_user_idx)

personal_mlp_batch_times_eager = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = personal_model(p_batch_embeddings, p_batch_user_idx)
        personal_mlp_batch_times_eager.append(time.time() - start_time)

personal_mlp_batch_fps_eager = (p_batch_embeddings.shape[0] * num_batches) / np.sum(personal_mlp_batch_times_eager)
print(f"Personalized MLP Batch Throughput (Eager, CPU, batch_size=32): {personal_mlp_batch_fps_eager:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Personalized MLP compiled mode (CPU)

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
personal_model = torch.compile(personal_model)

# Warm-up (triggers compilation)
print("Compiling Personalized MLP model (this may take a moment)...")
with torch.no_grad():
    personal_model(p_single_embedding, p_single_user_idx)
print("Compilation complete.")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
personal_mlp_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = personal_model(p_single_embedding, p_single_user_idx)
        personal_mlp_latencies_compiled.append(time.time() - start_time)

print("Personalized MLP Single Sample Latency (Compiled, CPU):")
print(f"  Median: {np.percentile(personal_mlp_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(personal_mlp_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(personal_mlp_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(personal_mlp_latencies_compiled):.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
personal_mlp_batch_times_compiled = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = personal_model(p_batch_embeddings, p_batch_user_idx)
        personal_mlp_batch_times_compiled.append(time.time() - start_time)

personal_mlp_batch_fps_compiled = (p_batch_embeddings.shape[0] * num_batches) / np.sum(personal_mlp_batch_times_compiled)
print(f"Personalized MLP Batch Throughput (Compiled, CPU, batch_size=32): {personal_mlp_batch_fps_compiled:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Personalized MLP CPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 65)
print("Personalized MLP Head CPU Benchmark Summary")
print("=" * 65)
print(f"Personalized MLP Model Size on Disk: {personal_mlp_model_size / (1e6):.2f} MB")
print()
print(f"{'Metric':<45} {'Eager':>8} {'Compiled':>8}")
print("-" * 65)
print(f"{'Single sample latency (median, ms)':<45} {np.percentile(personal_mlp_latencies_eager, 50)*1000:>8.2f} {np.percentile(personal_mlp_latencies_compiled, 50)*1000:>8.2f}")
print(f"{'Single sample latency (p95, ms)':<45} {np.percentile(personal_mlp_latencies_eager, 95)*1000:>8.2f} {np.percentile(personal_mlp_latencies_compiled, 95)*1000:>8.2f}")
print(f"{'Single sample throughput (FPS)':<45} {num_trials/np.sum(personal_mlp_latencies_eager):>8.2f} {num_trials/np.sum(personal_mlp_latencies_compiled):>8.2f}")
print(f"{'Batch throughput (FPS, batch_size=32)':<45} {personal_mlp_batch_fps_eager:>8.2f} {personal_mlp_batch_fps_compiled:>8.2f}")
```
:::



::: {.cell .markdown}

---

## Part 4: End-to-End Pipeline (CPU)

Finally, let's measure the full pipeline: image → ViT → normalize → MLP → score. This shows the total latency a user would experience. We'll reload fresh (uncompiled) models.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload uncompiled models for E2E measurement
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()
personal_model = torch.load(personal_model_path, map_location=device, weights_only=False)
personal_model.eval()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 50

# Single image E2E
single_image = single_image[:1].to(device)

# Warm-up
with torch.no_grad():
    feat = clip_model.encode_image(single_image)
    emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
    model(emb)

monitor.start()
e2e_latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = model(emb)
        e2e_latencies.append(time.time() - start_time)
monitor.stop()

print("End-to-End Single Image Latency (CPU):")
print(f"  Median: {np.percentile(e2e_latencies, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_latencies, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_latencies):.2f} FPS")
monitor.summary("E2E pipeline single image (CPU)")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Batch E2E
batch_images, _ = next(iter(test_loader))
batch_images = batch_images.to(device)

e2e_batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = model(emb)
        e2e_batch_times.append(time.time() - start_time)

e2e_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_batch_times)
print(f"End-to-End Batch Throughput (CPU, batch_size=32): {e2e_batch_fps:.2f} FPS")
```
:::

::: {.cell .markdown}

#### End-to-End: Personalized MLP (CPU)

The personalized pipeline adds a user index lookup, but the ViT encode step is identical.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Single image E2E - Personalized
p_user_idx_single = torch.tensor([0], dtype=torch.long, device=device)

# Warm-up
with torch.no_grad():
    feat = clip_model.encode_image(single_image)
    emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
    personal_model(emb, p_user_idx_single)

e2e_personal_latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = personal_model(emb, p_user_idx_single)
        e2e_personal_latencies.append(time.time() - start_time)

print("End-to-End Single Image Latency - Personalized (CPU):")
print(f"  Median: {np.percentile(e2e_personal_latencies, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_personal_latencies, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_personal_latencies):.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Batch E2E - Personalized
p_user_idx_batch = torch.zeros(batch_images.shape[0], dtype=torch.long, device=device)

e2e_personal_batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = personal_model(emb, p_user_idx_batch)
        e2e_personal_batch_times.append(time.time() - start_time)

e2e_personal_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_personal_batch_times)
print(f"End-to-End Batch Throughput - Personalized (CPU, batch_size=32): {e2e_personal_batch_fps:.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Latency breakdown
vit_median = np.percentile(vit_latencies_eager, 50) * 1000
mlp_median = np.percentile(mlp_latencies_eager, 50) * 1000
personal_mlp_median = np.percentile(personal_mlp_latencies_eager, 50) * 1000
e2e_median = np.percentile(e2e_latencies, 50) * 1000
e2e_personal_median = np.percentile(e2e_personal_latencies, 50) * 1000

print("=" * 55)
print("End-to-End Latency Breakdown (CPU, single image)")
print("=" * 55)
print(f"  ViT encode:            {vit_median:.2f} ms ({vit_median/e2e_median*100:.1f}%)")
print(f"  Global MLP forward:    {mlp_median:.2f} ms ({mlp_median/e2e_median*100:.1f}%)")
print(f"  Personal MLP forward:  {personal_mlp_median:.2f} ms ({personal_mlp_median/e2e_personal_median*100:.1f}%)")
print(f"  E2E total (global):    {e2e_median:.2f} ms")
print(f"  E2E total (personal):  {e2e_personal_median:.2f} ms")
print()
print("The ViT encoder dominates the pipeline cost.")
print("Optimizing the MLP (ONNX, quantization, etc.) will")
print("improve MLP latency, but the total pipeline speedup")
print("depends primarily on the ViT encoder performance.")
```
:::

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::
