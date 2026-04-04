
::: {.cell .markdown}

## Measure inference performance of PyTorch model on GPU 

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

### Model architecture definitions

The `.pth` files contain saved state dicts (model weights only). We need to define the same architectures used during training before loading the weights.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import torch.nn as nn

class GlobalMLP(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class PersonalizedMLP(nn.Module):
    def __init__(self, num_users, input_dim=768, user_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + user_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, user_idx):
        u = self.user_embedding(user_idx)
        z = torch.cat([x, u], dim=-1)
        return torch.sigmoid(self.net(z))
```
:::


::: {.cell .markdown}

First, let's load our MLP head and the CLIP ViT-L/14 model (used to compute image embeddings). We run all inference on the **GPU** — the A100's tensor cores make ViT encoding orders of magnitude faster than CPU.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_path = "models/inference_only/flickr_global_best_inference_only.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = GlobalMLP()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.to(device)
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

## Part 1: CLIP ViT-L/14 Image Encoder (GPU)

The ViT-L/14 model is the computationally expensive part of the pipeline. It processes raw images (224×224) through a Vision Transformer to produce 768-dimensional embeddings. Let's measure its performance on GPU in **eager mode** first.

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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        clip_model.encode_image(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_i = time.time() - start_time
        vit_latencies_eager.append(latencies_i)
monitor.stop()

print("ViT-L/14 Single Image Latency (Eager, GPU):")
print(f"  Median: {np.percentile(vit_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(vit_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(vit_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(vit_latencies_eager):.2f} FPS")
monitor.summary("ViT-L/14 eager single image (GPU)")
```
:::

::: {.cell .markdown}

#### ViT batch throughput (eager mode)

We sweep batch sizes from 32 up to 1024 to see how throughput and GPU utilization scale. We load 1024 images once and slice to each batch size for a fair comparison across sizes.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
batch_sizes = [32, 64, 128, 256, 512, 1024]
num_batches = 50

# Load 1024 images once — slice to each batch size for fair comparison
big_loader = DataLoader(test_dataset, batch_size=max(batch_sizes), shuffle=False, num_workers=4)
all_images, _ = next(iter(big_loader))
all_images = all_images.to(device)
print(f"Loaded {all_images.shape[0]} images for batch sweep")

vit_batch_results_eager = {}
for bs in batch_sizes:
    batch = all_images[:bs]

    # Warm-up
    with torch.no_grad():
        clip_model.encode_image(batch)

    monitor.start()
    times = []
    with torch.no_grad():
        for _ in range(num_batches):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            clip_model.encode_image(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start_time)
    monitor.stop()

    fps = (bs * num_batches) / np.sum(times)
    vit_batch_results_eager[bs] = {"times": times, "fps": fps,
                                   "gpu_util": np.mean(monitor.gpu_util) if monitor.gpu_util else 0,
                                   "gpu_mem": max(monitor.gpu_mem_used) if monitor.gpu_mem_used else 0}

    print(f"\nbatch_size={bs}:")
    print(f"  Median: {np.percentile(times, 50) * 1000:.2f} ms")
    print(f"  95th percentile: {np.percentile(times, 95) * 1000:.2f} ms")
    print(f"  Throughput: {fps:.2f} FPS")
    print(f"  GPU util: {vit_batch_results_eager[bs]['gpu_util']:.1f}%  "
          f"GPU mem peak: {vit_batch_results_eager[bs]['gpu_mem']:.0f} MB")
    monitor.summary(f"ViT-L/14 eager batch_size={bs} (GPU)")
```
:::


::: {.cell .markdown}

#### ViT compiled mode (GPU)

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
monitor.start()
vit_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        clip_model.encode_image(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vit_latencies_compiled.append(time.time() - start_time)
monitor.stop()

print("ViT-L/14 Single Image Latency (Compiled, GPU):")
print(f"  Median: {np.percentile(vit_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(vit_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(vit_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(vit_latencies_compiled):.2f} FPS")
monitor.summary("ViT-L/14 compiled single image (GPU)")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Batch throughput (compiled) — sweep same batch sizes
vit_batch_results_compiled = {}
for bs in batch_sizes:
    batch = all_images[:bs]

    # Warm-up (compiled graph may recompile for new shape)
    with torch.no_grad():
        clip_model.encode_image(batch)

    monitor.start()
    times = []
    with torch.no_grad():
        for _ in range(num_batches):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            clip_model.encode_image(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start_time)
    monitor.stop()

    fps = (bs * num_batches) / np.sum(times)
    vit_batch_results_compiled[bs] = {"times": times, "fps": fps,
                                      "gpu_util": np.mean(monitor.gpu_util) if monitor.gpu_util else 0,
                                      "gpu_mem": max(monitor.gpu_mem_used) if monitor.gpu_mem_used else 0}

    print(f"\nbatch_size={bs}:")
    print(f"  Median: {np.percentile(times, 50) * 1000:.2f} ms")
    print(f"  95th percentile: {np.percentile(times, 95) * 1000:.2f} ms")
    print(f"  Throughput: {fps:.2f} FPS")
    print(f"  GPU util: {vit_batch_results_compiled[bs]['gpu_util']:.1f}%  "
          f"GPU mem peak: {vit_batch_results_compiled[bs]['gpu_mem']:.0f} MB")
    monitor.summary(f"ViT-L/14 compiled batch_size={bs} (GPU)")
```
:::

::: {.cell .markdown}

#### ViT GPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 100)
print("CLIP ViT-L/14 GPU Benchmark Summary")
print("=" * 100)
print(f"{'Metric':<45} {'Eager':>10} {'Compiled':>10}")
print("-" * 70)
print(f"{'Single image latency (median, ms)':<45} {np.percentile(vit_latencies_eager, 50)*1000:>10.2f} {np.percentile(vit_latencies_compiled, 50)*1000:>10.2f}")
print(f"{'Single image latency (p95, ms)':<45} {np.percentile(vit_latencies_eager, 95)*1000:>10.2f} {np.percentile(vit_latencies_compiled, 95)*1000:>10.2f}")
print(f"{'Single image throughput (FPS)':<45} {num_trials/np.sum(vit_latencies_eager):>10.2f} {num_trials/np.sum(vit_latencies_compiled):>10.2f}")
print()
print(f"{'Batch size':<12} {'Eager FPS':>10} {'Compiled FPS':>13} {'Eager p50 ms':>13} {'Compiled p50 ms':>16} {'Eager mem MB':>13} {'Compiled mem MB':>16}")
print("-" * 100)
for bs in batch_sizes:
    e = vit_batch_results_eager[bs]
    c = vit_batch_results_compiled[bs]
    print(f"{bs:<12} {e['fps']:>10.1f} {c['fps']:>13.1f} {np.percentile(e['times'], 50)*1000:>13.2f} {np.percentile(c['times'], 50)*1000:>16.2f} {e['gpu_mem']:>13.0f} {c['gpu_mem']:>16.0f}")
```
:::


::: {.cell .markdown}

---

## Part 2: Aesthetic MLP Head (GPU)

Now we'll benchmark the lightweight MLP head that maps 768-dim CLIP embeddings to aesthetic scores. Since we'll compare eager vs compiled mode, we'll first reload the models fresh.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload CLIP (uncompiled) and MLP for clean benchmarking
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
model = GlobalMLP()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.to(device)
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(single_embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_latencies_eager.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("MLP Single Sample Latency (Eager, GPU):")
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(batch_embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_batch_times_eager.append(time.time() - start_time)

mlp_batch_fps_eager = (batch_embeddings.shape[0] * num_batches) / np.sum(mlp_batch_times_eager)
print(f"MLP Batch Throughput (Eager, GPU, batch_size=32): {mlp_batch_fps_eager:.2f} FPS")
```
:::


::: {.cell .markdown}

#### MLP compiled mode (GPU)

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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(single_embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_latencies_compiled.append(time.time() - start_time)

print("MLP Single Sample Latency (Compiled, GPU):")
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(batch_embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_batch_times_compiled.append(time.time() - start_time)

mlp_batch_fps_compiled = (batch_embeddings.shape[0] * num_batches) / np.sum(mlp_batch_times_compiled)
print(f"MLP Batch Throughput (Compiled, GPU, batch_size=32): {mlp_batch_fps_compiled:.2f} FPS")
```
:::


::: {.cell .markdown}

#### MLP GPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 60)
print("Aesthetic MLP Head GPU Benchmark Summary")
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

## Part 3: Personalized MLP Head (GPU)

The personalized model takes both a 768-dim CLIP embedding and a user index as input. It has an `nn.Embedding` table that maps each user index to a 64-dim learned vector, concatenates it with the CLIP embedding (832-dim total), and passes through the same MLP architecture. This lets the model learn per-user aesthetic preferences.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload CLIP (uncompiled) for clean benchmarking
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Load personalized model
personal_model_path = "models/inference_only/flickr_personalized_best_inference_only.pth"
_p_state = torch.load(personal_model_path, map_location=device, weights_only=False)
_num_users = _p_state["user_embedding.weight"].shape[0]
personal_model = PersonalizedMLP(num_users=_num_users)
personal_model.load_state_dict(_p_state)
personal_model.to(device)
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_single_embedding, p_single_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_latencies_eager.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Personalized MLP Single Sample Latency (Eager, GPU):")
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_batch_embeddings, p_batch_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_batch_times_eager.append(time.time() - start_time)

personal_mlp_batch_fps_eager = (p_batch_embeddings.shape[0] * num_batches) / np.sum(personal_mlp_batch_times_eager)
print(f"Personalized MLP Batch Throughput (Eager, GPU, batch_size=32): {personal_mlp_batch_fps_eager:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Personalized MLP compiled mode (GPU)

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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_single_embedding, p_single_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_latencies_compiled.append(time.time() - start_time)

print("Personalized MLP Single Sample Latency (Compiled, GPU):")
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_batch_embeddings, p_batch_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_batch_times_compiled.append(time.time() - start_time)

personal_mlp_batch_fps_compiled = (p_batch_embeddings.shape[0] * num_batches) / np.sum(personal_mlp_batch_times_compiled)
print(f"Personalized MLP Batch Throughput (Compiled, GPU, batch_size=32): {personal_mlp_batch_fps_compiled:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Personalized MLP GPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 65)
print("Personalized MLP Head GPU Benchmark Summary")
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

## Part 4: End-to-End Pipeline (GPU)

Finally, let's measure the full pipeline: image → ViT → normalize → MLP → score. This shows the total latency a user would experience. We'll reload fresh (uncompiled) models.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload uncompiled models for E2E measurement
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
model = GlobalMLP()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.to(device)
model.eval()
personal_model = PersonalizedMLP(num_users=_num_users)
personal_model.load_state_dict(torch.load(personal_model_path, map_location=device, weights_only=False))
personal_model.to(device)
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = model(emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_latencies.append(time.time() - start_time)
monitor.stop()

print("End-to-End Single Image Latency (GPU):")
print(f"  Median: {np.percentile(e2e_latencies, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_latencies, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_latencies):.2f} FPS")
monitor.summary("E2E pipeline single image (GPU)")
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = model(emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_batch_times.append(time.time() - start_time)

e2e_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_batch_times)
print(f"End-to-End Batch Throughput (GPU, batch_size=32): {e2e_batch_fps:.2f} FPS")
```
:::

::: {.cell .markdown}

#### End-to-End: Personalized MLP (GPU)

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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = personal_model(emb, p_user_idx_single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_personal_latencies.append(time.time() - start_time)

print("End-to-End Single Image Latency - Personalized (GPU):")
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = personal_model(emb, p_user_idx_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_personal_batch_times.append(time.time() - start_time)

e2e_personal_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_personal_batch_times)
print(f"End-to-End Batch Throughput - Personalized (GPU, batch_size=32): {e2e_personal_batch_fps:.2f} FPS")
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
print("End-to-End Latency Breakdown (GPU, single image)")
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

---

## Part 5: Quality Metrics

Performance benchmarks tell you *how fast* the models run. Quality metrics tell you *how well* they predict — essential for right-sizing decisions (no point in ultra-low latency on a model that lacks accuracy).

We run both models over their respective held-out inference sets and compute:

**Global MLP**: MAE, RMSE, PLCC, SRCC, binary accuracy (threshold = 0.5), AUC-ROC  
**Personalized MLP**: same per-user metrics averaged across users, plus personalization gain vs global

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from PIL import Image
import glob

# ── helpers ──────────────────────────────────────────────────────────────────

def collect_global_predictions(manifest_df, image_root, clip_model, mlp_model, device,
                               clip_preprocess, batch_size=64):
    """Run the full ViT → MLP pipeline over all images in manifest_df.
    Returns (preds, targets) as numpy arrays."""
    preprocess = clip_preprocess
    preds, targets = [], []
    rows = manifest_df.reset_index(drop=True)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start + batch_size]
        imgs = []
        valid_mask = []
        for _, row in batch_rows.iterrows():
            img_path = os.path.join(image_root, row["image_name"])
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                imgs.append(img)
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
        if not imgs:
            continue
        img_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(img_tensor)
            embs = torch.from_numpy(normalized(feats.cpu().numpy())).float().to(device)
            scores = mlp_model(embs).squeeze().cpu().numpy()
        if scores.ndim == 0:
            scores = scores.reshape(1)
        gt = batch_rows.loc[[v for v, m in zip(batch_rows.index, valid_mask) if m], "global_score"].values
        preds.extend(scores.tolist())
        targets.extend(gt.tolist())

    return np.array(preds), np.array(targets)


def print_regression_metrics(preds, targets, label=""):
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    plcc, _ = pearsonr(preds, targets)
    srcc, _ = spearmanr(preds, targets)
    threshold = 0.5
    bin_acc = np.mean((preds >= threshold) == (targets >= threshold))
    try:
        auc = roc_auc_score((targets >= threshold).astype(int), preds)
    except ValueError:
        auc = float("nan")  # only one class present

    print(f"\n{'─'*55}")
    print(f"Quality metrics — {label}")
    print(f"{'─'*55}")
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  PLCC:             {plcc:.4f}")
    print(f"  SRCC:             {srcc:.4f}")
    print(f"  Binary accuracy:  {bin_acc:.4f}  (threshold={threshold})")
    print(f"  AUC-ROC:          {auc:.4f}")
    return dict(mae=mae, rmse=rmse, plcc=plcc, srcc=srcc, bin_acc=bin_acc, auc=auc)
```
:::

::: {.cell .markdown}

### Global MLP — quality metrics

Load the test split from `flickr_global_manifest.csv` and run inference over all held-out images.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
global_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_global_manifest.csv"))

# Use only the inference (test) split
global_test = global_manifest[global_manifest["split"] == "inference"].copy()
image_root_global = os.path.join(data_dir, "40K")

print(f"Global test set: {len(global_test)} images")
global_test.head()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload clean models (without compile artifacts)
model_eval = GlobalMLP()
model_eval.load_state_dict(torch.load("models/inference_only/flickr_global_best_inference_only.pth",
                        map_location=device, weights_only=False))
model_eval.to(device)
model_eval.eval()
clip_eval, clip_pre_eval = clip.load("ViT-L/14", device=device)

print(f"Running GPU CLIP encoding over test set (~1-3 minutes)...")
global_preds, global_targets = collect_global_predictions(
    global_test, image_root_global, clip_eval, model_eval, device, clip_pre_eval
)
print(f"Collected {len(global_preds)} predictions")
global_metrics = print_regression_metrics(global_preds, global_targets, label="Global MLP")
```
:::

::: {.cell .markdown}

### Personalized MLP — quality metrics

For the personalized model we compute metrics per user (using their held-out (image, score) pairs), then average across users.  
We also compute **personalization gain**: how much the personalized model improves on the global model for the same samples.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
personal_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))

# Use only the inference split
personal_test = personal_manifest[personal_manifest["split"] == "inference"].copy()

# Use worker_score_norm as ground truth (already in [0,1])
personal_test = personal_test.rename(columns={"worker_score_norm": "global_score"})

# Build user index mapping (must match training)
seen_workers = sorted(
    personal_manifest.loc[personal_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique()
)
user2idx = {u: i for i, u in enumerate(seen_workers)}

image_root_personal = os.path.join(data_dir, "40K")
_p_eval_state = torch.load("models/inference_only/flickr_personalized_best_inference_only.pth",
                                  map_location=device, weights_only=False)
personal_model_eval = PersonalizedMLP(num_users=_p_eval_state["user_embedding.weight"].shape[0])
personal_model_eval.load_state_dict(_p_eval_state)
personal_model_eval.to(device)
personal_model_eval.eval()

print(f"Personalized test set: {len(personal_test)} rows, "
      f"{personal_test['worker_id'].nunique()} users")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
per_user_srcc = []
per_user_mae  = []
all_personal_preds   = []
all_personal_targets = []
all_global_preds_on_personal = []

preprocess = clip_pre_eval
test_workers = [w for w in personal_test["worker_id"].unique() if w in user2idx]

for worker_id in test_workers:
    user_rows = personal_test[personal_test["worker_id"] == worker_id].reset_index(drop=True)
    if len(user_rows) < 3:
        continue  # skip users with too few samples for meaningful correlation
    uid = user2idx[worker_id]

    imgs, gt_scores = [], []
    for _, row in user_rows.iterrows():
        img_path = os.path.join(image_root_personal, row["image_name"])
        try:
            imgs.append(preprocess(Image.open(img_path).convert("RGB")))
            gt_scores.append(row["global_score"])
        except Exception:
            pass
    if not imgs:
        continue

    img_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = clip_eval.encode_image(img_tensor)
        embs  = torch.from_numpy(normalized(feats.cpu().numpy())).float().to(device)
        user_idx_tensor = torch.full((len(imgs),), uid, dtype=torch.long, device=device)
        p_scores = personal_model_eval(embs, user_idx_tensor).squeeze().cpu().numpy()
        g_scores = model_eval(embs).squeeze().cpu().numpy()

    if p_scores.ndim == 0:
        p_scores = p_scores.reshape(1)
    if g_scores.ndim == 0:
        g_scores = g_scores.reshape(1)

    gt = np.array(gt_scores)
    srcc_u, _ = spearmanr(p_scores, gt)
    mae_u     = np.mean(np.abs(p_scores - gt))
    per_user_srcc.append(srcc_u)
    per_user_mae.append(mae_u)
    all_personal_preds.extend(p_scores.tolist())
    all_personal_targets.extend(gt.tolist())
    all_global_preds_on_personal.extend(g_scores.tolist())

all_personal_preds          = np.array(all_personal_preds)
all_personal_targets        = np.array(all_personal_targets)
all_global_preds_on_personal = np.array(all_global_preds_on_personal)

print(f"Evaluated {len(test_workers)} users")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Aggregate metrics
personal_metrics = print_regression_metrics(
    all_personal_preds, all_personal_targets, label="Personalized MLP (all users pooled)"
)

print(f"\n  Per-user SRCC (avg):  {np.mean(per_user_srcc):.4f}  "
      f"(std={np.std(per_user_srcc):.4f})")
print(f"  Per-user MAE  (avg):  {np.mean(per_user_mae):.4f}  "
      f"(std={np.std(per_user_mae):.4f})")

# Personalization gain
global_mae_on_personal = np.mean(np.abs(all_global_preds_on_personal - all_personal_targets))
personal_mae_on_personal = personal_metrics["mae"]
gain_mae = global_mae_on_personal - personal_mae_on_personal

global_srcc_on_personal, _ = spearmanr(all_global_preds_on_personal, all_personal_targets)
personal_srcc_on_personal, _ = spearmanr(all_personal_preds, all_personal_targets)
gain_srcc = personal_srcc_on_personal - global_srcc_on_personal

print(f"\n{'─'*55}")
print(f"Personalization gain (personalized vs global, same images)")
print(f"{'─'*55}")
print(f"  Global MAE  on personal set:  {global_mae_on_personal:.4f}")
print(f"  Personal MAE on personal set: {personal_mae_on_personal:.4f}")
print(f"  MAE improvement:              {gain_mae:+.4f}  {'✓ better' if gain_mae > 0 else '✗ worse'}")
print(f"  Global SRCC:                  {global_srcc_on_personal:.4f}")
print(f"  Personal SRCC:                {personal_srcc_on_personal:.4f}")
print(f"  SRCC improvement:             {gain_srcc:+.4f}  {'✓ better' if gain_srcc > 0 else '✗ worse'}")
```
:::

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::
