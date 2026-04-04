

::: {.cell .markdown}

### GPU Inference: ViT Encoder and ONNX Execution Providers

In this notebook, we will:

1. Benchmark the **CLIP ViT-L/14 image encoder on GPU** (eager + compiled, across multiple batch sizes)
2. Measure the **end-to-end pipeline on GPU** (image → ViT → MLP → score)
3. Test the **MLP head with different ONNX execution providers** (CPU, CUDA, TensorRT, OpenVINO)

You are already running in the `jupyter-onnx-gpu` container that was launched in notebook 4 — no container switch is needed for Parts 1, 2, and 3 (CUDA and TensorRT execution providers).

> **Note**: The OpenVINO execution provider requires a separate container. Instructions for switching are provided in the OpenVINO section later in this notebook.

:::



::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
import clip
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from PIL import Image
```
:::

::: {.cell .markdown}

## Resource monitoring

The `ResourceMonitor` class polls `nvidia-smi` (GPU utilization and memory) and `psutil` (CPU and RAM) in a background thread. It runs alongside each execution provider benchmark so you can see exactly how much GPU memory and compute each EP consumes — the primary signal for right-sizing.

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

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load CLIP model on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# Prepare test dataset with CLIP preprocessing
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
```
:::



::: {.cell .markdown}

---

## Part 1: CLIP ViT-L/14 on GPU

The ViT is the heavy part of the pipeline. On GPU, we can process much larger batches efficiently. We'll benchmark across multiple batch sizes to see how throughput scales and find the point where the GPU is fully utilized.

:::

::: {.cell .markdown}

#### ViT GPU: Eager mode across batch sizes

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
batch_sizes = [1, 8, 32, 64, 128, 256, 512]
num_trials_single = 100
num_batches_multi = 50

vit_gpu_eager_results = {}

for bs in batch_sizes:
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)
    images, _ = next(iter(loader))
    images = images.to(device)

    # Warm-up
    with torch.no_grad():
        clip_model.encode_image(images)
    if device.type == "cuda":
        torch.cuda.synchronize()

    trials = num_trials_single if bs == 1 else num_batches_multi
    latencies = []
    with torch.no_grad():
        for _ in range(trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            clip_model.encode_image(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.time() - start_time)

    median_ms = np.percentile(latencies, 50) * 1000
    p95_ms = np.percentile(latencies, 95) * 1000
    fps = (bs * trials) / np.sum(latencies)

    vit_gpu_eager_results[bs] = {
        'median_ms': median_ms, 'p95_ms': p95_ms, 'fps': fps, 'latencies': latencies
    }
    print(f"  batch_size={bs:>3}: median={median_ms:.2f} ms, p95={p95_ms:.2f} ms, throughput={fps:.1f} FPS")
```
:::


::: {.cell .markdown}

#### ViT GPU: Compiled mode across batch sizes

Now let's compile the ViT visual encoder for potential further speedup on GPU.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Compiling ViT visual encoder for GPU...")
clip_model.visual = torch.compile(clip_model.visual)

# Warm-up with compilation (uses batch_size=32 to trigger compilation)
loader_32 = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
warmup_images, _ = next(iter(loader_32))
with torch.no_grad():
    clip_model.encode_image(warmup_images.to(device))
if device.type == "cuda":
    torch.cuda.synchronize()
print("Compilation complete.")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
vit_gpu_compiled_results = {}

for bs in batch_sizes:
    loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)
    images, _ = next(iter(loader))
    images = images.to(device)

    # Warm-up
    with torch.no_grad():
        clip_model.encode_image(images)
    if device.type == "cuda":
        torch.cuda.synchronize()

    trials = num_trials_single if bs == 1 else num_batches_multi
    latencies = []
    with torch.no_grad():
        for _ in range(trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            clip_model.encode_image(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.time() - start_time)

    median_ms = np.percentile(latencies, 50) * 1000
    p95_ms = np.percentile(latencies, 95) * 1000
    fps = (bs * trials) / np.sum(latencies)

    vit_gpu_compiled_results[bs] = {
        'median_ms': median_ms, 'p95_ms': p95_ms, 'fps': fps, 'latencies': latencies
    }
    print(f"  batch_size={bs:>3}: median={median_ms:.2f} ms, p95={p95_ms:.2f} ms, throughput={fps:.1f} FPS")
```
:::

::: {.cell .markdown}

#### ViT GPU summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("=" * 75)
print("CLIP ViT-L/14 GPU Benchmark Summary")
print("=" * 75)
print(f"{'Batch Size':>10} | {'Eager (ms)':>11} {'Eager FPS':>10} | {'Compiled (ms)':>14} {'Compiled FPS':>13}")
print("-" * 75)
for bs in batch_sizes:
    e = vit_gpu_eager_results[bs]
    c = vit_gpu_compiled_results[bs]
    print(f"{bs:>10} | {e['median_ms']:>11.2f} {e['fps']:>10.1f} | {c['median_ms']:>14.2f} {c['fps']:>13.1f}")
```
:::


::: {.cell .markdown}

---

## Part 2: End-to-End Pipeline on GPU

Let's measure the full pipeline on GPU: image → ViT (GPU) → normalize → MLP (CPU) → score. The MLP stays on CPU since it's tiny and the overhead of GPU kernel launch would likely dominate.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Reload uncompiled CLIP for clean E2E measurement
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Load MLP on CPU
mlp_model = GlobalMLP()
mlp_model.load_state_dict(torch.load("models/inference_only/flickr_global_best_inference_only.pth", map_location=torch.device("cpu"), weights_only=False))
mlp_model.eval()

num_trials = 50

# Single image E2E
loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
single_image, _ = next(iter(loader))
single_image = single_image.to(device)

# Warm-up
with torch.no_grad():
    feat = clip_model.encode_image(single_image)
    emb = torch.from_numpy(normalized(feat.cpu().numpy())).float()
    mlp_model(emb)

e2e_latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float()
        _ = mlp_model(emb)
        e2e_latencies.append(time.time() - start_time)

print("End-to-End Single Image (ViT on GPU, MLP on CPU):")
print(f"  Median: {np.percentile(e2e_latencies, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_latencies, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_latencies):.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Batch E2E (batch_size=32)
loader_32 = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
batch_images, _ = next(iter(loader_32))
batch_images = batch_images.to(device)

num_batches = 50
e2e_batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float()
        _ = mlp_model(emb)
        e2e_batch_times.append(time.time() - start_time)

e2e_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_batch_times)
print(f"End-to-End Batch (batch_size=32, ViT on GPU, MLP on CPU): {e2e_batch_fps:.2f} FPS")
```
:::


::: {.cell .markdown}

### Personalized MLP: End-to-End Pipeline

We repeat the end-to-end measurement with the **PersonalizedMLP**, which takes an additional **user index** input alongside the CLIP embedding.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load personalized model and prepare user indices
_p_state = torch.load("models/inference_only/flickr_personalized_best_inference_only.pth", map_location="cpu", weights_only=False)
_num_users = _p_state["user_embedding.weight"].shape[0]
_user_dim  = _p_state["user_embedding.weight"].shape[1]
personal_model = PersonalizedMLP(num_users=_num_users, user_dim=_user_dim)
personal_model.load_state_dict(_p_state)
personal_model.eval()

manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
seen_workers = sorted(manifest[manifest["worker_split"] == "seen_worker_pool"]["worker_id"].unique())
user2idx = {u: i for i, u in enumerate(seen_workers)}
print(f"Personalized model loaded. Number of seen users: {len(seen_workers)}")

# Use first user for benchmarking
sample_user_idx = torch.tensor([0], dtype=torch.long)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Personalized End-to-End single image latency
num_trials = 50

with torch.no_grad():
    # Warm-up
    feat = clip_model.encode_image(single_image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    emb = torch.from_numpy(normalized(feat.cpu().numpy())).float()
    _ = personal_model(emb, sample_user_idx)

    e2e_personal_times = []
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float()
        _ = personal_model(emb, sample_user_idx)
        e2e_personal_times.append(time.time() - start_time)

print("Personalized E2E Single Image (ViT on GPU, MLP on CPU):")
print(f"  Median: {np.percentile(e2e_personal_times, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_personal_times, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_personal_times):.2f} FPS")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Personalized End-to-End batch throughput
batch_user_idx = torch.zeros(batch_images.shape[0], dtype=torch.long)
num_batches = 50

e2e_personal_batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float()
        _ = personal_model(emb, batch_user_idx)
        e2e_personal_batch_times.append(time.time() - start_time)

e2e_personal_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_personal_batch_times)
print(f"Personalized E2E Batch (batch_size=32, ViT on GPU, MLP on CPU): {e2e_personal_batch_fps:.2f} FPS")
```
:::


::: {.cell .markdown}

---

## Part 3: MLP ONNX Execution Providers

Now we'll benchmark the MLP head using different ONNX Runtime execution providers. Since the MLP takes pre-computed 768-dim embeddings as input, we pre-compute those once and then time only the ONNX inference.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Pre-compute CLIP embeddings for MLP ONNX benchmarking
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    batch_embeddings = normalized(batch_features.cpu().numpy()).astype(np.float32)
    single_embedding = batch_embeddings[:1]

# Prepare personalized inputs for ONNX benchmarking
user_idx_single = np.array([0], dtype=np.int64)
user_idx_batch = np.zeros(batch_embeddings.shape[0], dtype=np.int64)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
def benchmark_session(ort_session):

    print(f"Execution provider: {ort_session.get_providers()}")

    ## Sample predictions

    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: batch_embeddings})[0]
    scores = outputs.flatten()
    print(f"Sample scores (first 5): {', '.join(f'{s:.2f}' for s in scores[:5])}")
    print(f"Mean predicted score: {scores.mean():.2f}, Std: {scores.std():.2f}")

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_embedding})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_embedding})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput

    num_batches = 50  # Number of trials

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_embeddings})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: batch_embeddings})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_embeddings.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")

```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
def benchmark_personal_session(ort_session):

    input_names = [inp.name for inp in ort_session.get_inputs()]
    print(f"Execution provider: {ort_session.get_providers()}")

    ## Sample predictions

    outputs = ort_session.run(None, {input_names[0]: batch_embeddings, input_names[1]: user_idx_batch})[0]
    scores = outputs.flatten()
    print(f"Sample scores (first 5): {', '.join(f'{s:.2f}' for s in scores[:5])}")
    print(f"Mean predicted score: {scores.mean():.2f}, Std: {scores.std():.2f}")

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials

    # Warm-up run
    ort_session.run(None, {input_names[0]: single_embedding, input_names[1]: user_idx_single})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {input_names[0]: single_embedding, input_names[1]: user_idx_single})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput

    num_batches = 50  # Number of trials

    # Warm-up run
    ort_session.run(None, {input_names[0]: batch_embeddings, input_names[1]: user_idx_batch})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {input_names[0]: batch_embeddings, input_names[1]: user_idx_batch})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_embeddings.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")

```
:::




::: {.cell .markdown} 


#### CPU execution provider

First, for reference, we'll run the MLP ONNX model with the `CPUExecutionProvider`:

:::




::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Personalized MLP - CPU execution provider
personal_onnx_path = "models/flickr_personalized.onnx"
ort_session = ort.InferenceSession(personal_onnx_path, providers=['CPUExecutionProvider'])
benchmark_personal_session(ort_session)
```
:::

<!-- placeholder: update with real benchmark numbers -->


::: {.cell .markdown} 

#### CUDA execution provider

Next, we'll try the CUDA execution provider, which will execute the MLP model on the GPU:

:::




::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
monitor.start()
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
benchmark_session(ort_session)
monitor.stop()
monitor.summary("Global MLP — CUDAExecutionProvider")
ort.get_device()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Personalized MLP - CUDA execution provider
personal_onnx_path = "models/flickr_personalized.onnx"
monitor.start()
ort_session = ort.InferenceSession(personal_onnx_path, providers=['CUDAExecutionProvider'])
benchmark_personal_session(ort_session)
monitor.stop()
monitor.summary("Personalized MLP — CUDAExecutionProvider")
ort.get_device()
```
:::

<!-- placeholder: update with real benchmark numbers -->


::: {.cell .markdown}

#### Pre-compute test embeddings for TRT precision check

TensorRT on Ampere GPUs may silently apply FP16 precision, which can degrade prediction quality. We pre-compute all test embeddings once on GPU (fast: seconds), then verify quality metrics after each TRT benchmark call.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Pre-compute all test embeddings on GPU (one-time; used for TRT quality checks below)
print("Pre-computing test embeddings on GPU for TRT quality verification...")
_trt_g_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_global_manifest.csv"))
_trt_test_g = _trt_g_manifest[_trt_g_manifest["split"] == "test"].reset_index(drop=True)
_trt_img_root = os.path.join(data_dir, "40K")

_trt_g_embs_list, _trt_g_tgts = [], []
with torch.no_grad():
    for _i in range(0, len(_trt_test_g), 64):
        _batch = _trt_test_g.iloc[_i:_i+64]
        _imgs, _tgts = [], []
        for _, _row in _batch.iterrows():
            try:
                _imgs.append(clip_preprocess(Image.open(os.path.join(_trt_img_root, _row["image_name"])).convert("RGB")))
                _tgts.append(_row["global_score"])
            except Exception:
                pass
        if not _imgs:
            continue
        _feats = clip_model.encode_image(torch.stack(_imgs).to(device))
        _trt_g_embs_list.append(normalized(_feats.cpu().numpy()).astype(np.float32))
        _trt_g_tgts.extend(_tgts)
_trt_g_embs = np.concatenate(_trt_g_embs_list, axis=0)
_trt_g_tgts = np.array(_trt_g_tgts, dtype=np.float32)

_trt_p_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
_trt_test_p = _trt_p_manifest[_trt_p_manifest["split"] == "test"].reset_index(drop=True)
_trt_seen_w = sorted(_trt_p_manifest.loc[_trt_p_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique())
_trt_user2idx = {u: i for i, u in enumerate(_trt_seen_w)}

_trt_p_embs_list, _trt_p_tgts, _trt_p_uidxs = [], [], []
with torch.no_grad():
    for _i in range(0, len(_trt_test_p), 64):
        _batch = _trt_test_p.iloc[_i:_i+64]
        _imgs, _tgts, _uids = [], [], []
        for _, _row in _batch.iterrows():
            if _row["worker_id"] not in _trt_user2idx:
                continue
            try:
                _imgs.append(clip_preprocess(Image.open(os.path.join(_trt_img_root, _row["image_name"])).convert("RGB")))
                _tgts.append(_row["worker_score_norm"])
                _uids.append(_trt_user2idx[_row["worker_id"]])
            except Exception:
                pass
        if not _imgs:
            continue
        _feats = clip_model.encode_image(torch.stack(_imgs).to(device))
        _trt_p_embs_list.append(normalized(_feats.cpu().numpy()).astype(np.float32))
        _trt_p_tgts.extend(_tgts)
        _trt_p_uidxs.extend(_uids)
_trt_p_embs = np.concatenate(_trt_p_embs_list, axis=0)
_trt_p_tgts  = np.array(_trt_p_tgts,  dtype=np.float32)
_trt_p_uidxs = np.array(_trt_p_uidxs, dtype=np.int64)
print(f"Ready: {len(_trt_g_embs)} global + {len(_trt_p_embs)} personalized test embeddings.")
```
:::


::: {.cell .markdown} 

#### TensorRT execution provider


The TensorRT execution provider will optimize the model for inference on NVIDIA GPUs. It will take a long time to run this cell, because it spends a lot of time optimizing the model (finding the best subgraphs, etc.) - but once the model is loaded, its inference time will be much faster than any of our previous tests.


:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
monitor.start()
ort_session = ort.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider'])
benchmark_session(ort_session)
monitor.stop()
monitor.summary("Global MLP — TensorrtExecutionProvider")
ort.get_device()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Quality metrics: Global MLP with TensorRT EP
# TRT on Ampere may use FP16; if these values degrade vs CUDA EP, precision loss is the cause.
_trt_g_preds = ort_session.run(None, {ort_session.get_inputs()[0].name: _trt_g_embs})[0].flatten()
_trt_mae  = np.mean(np.abs(_trt_g_preds - _trt_g_tgts))
_trt_rmse = np.sqrt(np.mean((_trt_g_preds - _trt_g_tgts) ** 2))
_trt_plcc, _ = pearsonr(_trt_g_preds, _trt_g_tgts)
_trt_srcc, _ = spearmanr(_trt_g_preds, _trt_g_tgts)
_trt_acc  = np.mean((_trt_g_preds >= 0.5) == (_trt_g_tgts >= 0.5))
_trt_auc  = roc_auc_score((_trt_g_tgts >= 0.5).astype(int), _trt_g_preds)
print(f"\n{'─'*60}")
print(f"Quality metrics — Global MLP | TensorrtExecutionProvider")
print(f"{'─'*60}")
print(f"  N:                {len(_trt_g_preds)}")
print(f"  MAE:              {_trt_mae:.4f}")
print(f"  RMSE:             {_trt_rmse:.4f}")
print(f"  PLCC:             {_trt_plcc:.4f}")
print(f"  SRCC:             {_trt_srcc:.4f}")
print(f"  Binary accuracy:  {_trt_acc:.4f}  (threshold=0.5)")
print(f"  AUC-ROC:          {_trt_auc:.4f}")
print("Compare with FP32 CUDA EP metrics: any drop indicates TF32/FP16 precision trade-off.")
print("(On Ampere/A100, TRT defaults to TF32 for matmuls; FP16 requires explicit trt_fp16_enable=True.)")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Personalized MLP - TensorRT execution provider
personal_onnx_path = "models/flickr_personalized.onnx"
monitor.start()
ort_session = ort.InferenceSession(personal_onnx_path, providers=['TensorrtExecutionProvider'])
benchmark_personal_session(ort_session)
monitor.stop()
monitor.summary("Personalized MLP — TensorrtExecutionProvider")
ort.get_device()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Quality metrics: Personalized MLP with TensorRT EP — per-user SRCC and MAE
_trt_p_in = [i.name for i in ort_session.get_inputs()]
_trt_p_preds = ort_session.run(None, {_trt_p_in[0]: _trt_p_embs, _trt_p_in[1]: _trt_p_uidxs})[0].flatten()
_trt_per_srcc, _trt_per_mae = [], []
for uid in np.unique(_trt_p_uidxs):
    mask = _trt_p_uidxs == uid
    if mask.sum() < 3:
        continue
    _s, _ = spearmanr(_trt_p_preds[mask], _trt_p_tgts[mask])
    _trt_per_srcc.append(_s)
    _trt_per_mae.append(np.mean(np.abs(_trt_p_preds[mask] - _trt_p_tgts[mask])))
print(f"\n{'─'*60}")
print(f"Quality metrics — Personalized MLP | TensorrtExecutionProvider")
print(f"{'─'*60}")
print(f"  Users evaluated:    {len(_trt_per_srcc)}")
print(f"  Mean per-user SRCC: {np.mean(_trt_per_srcc):.4f}")
print(f"  Mean per-user MAE:  {np.mean(_trt_per_mae):.4f}")
print("Compare with FP32 CUDA EP metrics: any drop indicates TF32/FP16 precision trade-off.")
print("(On Ampere/A100, TRT defaults to TF32 for matmuls; FP16 requires explicit trt_fp16_enable=True.)")
```
:::


::: {.cell .markdown} 


#### OpenVINO execution provider

Even just on CPU, we can still use an optimized execution provider to improve inference performance. We will try out the Intel [OpenVINO](https://github.com/openvinotoolkit/openvino) execution provider. However, ONNX runtime can be built to support CUDA/TensorRT or OpenVINO, but not both at the same time, so we will need to bring up a new container.

Close this Jupyter server tab - you will reopen it shortly, with a new token.

Go back to your SSH session on "node-serve-model", and stop the current Jupyter server:

```bash
# runs on node-serve-model
docker stop jupyter
```

Build the OpenVINO image:

```bash
# runs on node-serve-model
docker build -t jupyter-onnx-openvino -f model-serving-nvidia/docker/Dockerfile.jupyter-onnx-openvino .
```

Then, launch a container with the OpenVINO image:

```bash
# runs on node-serve-model
docker run  -d --rm  -p 8888:8888 \
    --shm-size 16G \
    -v ~/model-serving-nvidia/workspace:/home/jovyan/work/ \
    -v aesthetic_data:/mnt/ \
    -e AESTHETIC_DATA_DIR=/mnt/flickr-aes \
    --name jupyter \
    jupyter-onnx-openvino
```

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access).

Run

```bash
# runs on node-serve-model
docker exec jupyter jupyter server list
```

and look for a line like

```
http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `8_ep_onnx.ipynb` notebook to continue.

Run the cells at the top, which `import` libraries, set up the data loaders, and define the `benchmark_session` function. Then, skip to the OpenVINO section and run:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['OpenVINOExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Personalized MLP - OpenVINO execution provider
personal_onnx_path = "models/flickr_personalized.onnx"
ort_session = ort.InferenceSession(personal_onnx_path, providers=['OpenVINOExecutionProvider'])
benchmark_personal_session(ort_session)
ort.get_device()
```
:::


<!-- placeholder: update with real benchmark numbers -->

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::
