

::: {.cell .markdown}

### GPU Inference: ViT Encoder and ONNX Execution Providers

In this notebook, we will:

1. Benchmark the **CLIP ViT-L/14 image encoder on GPU** (eager + compiled, across multiple batch sizes)
2. Measure the **end-to-end pipeline on GPU** (image → ViT → MLP → score)
3. Test the **MLP head with different ONNX execution providers** (CPU, CUDA, TensorRT, OpenVINO)

Before we can use the GPU, we need to switch from the `jupyter-onnx-base` image to the `jupyter-onnx-gpu` image.

Close the current Jupyter server tab - you will reopen it shortly, with a new token.

Go back to your SSH session on "node-serve-model", and stop the current Jupyter server with:

```bash
# runs on node-serve-model
docker stop jupyter
```

Build the GPU image:

```bash
# runs on node-serve-model
docker build -t jupyter-onnx-gpu -f model-serving-nvidia/docker/Dockerfile.jupyter-onnx-nvidia .
```

Then launch a new one with the GPU image:

```bash
# runs on node-serve-model
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/model-serving-nvidia/workspace:/home/jovyan/work/ \
    -v aesthetic_data:/mnt/ \
    -e AESTHETIC_DATA_DIR=/mnt/aesthetic-hub \
    --name jupyter \
    jupyter-onnx-gpu
```

Then get a new token:

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
import clip
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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# Prepare test dataset with CLIP preprocessing
data_dir = os.getenv("AESTHETIC_DATA_DIR", "aesthetic-hub")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=clip_preprocess)
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
mlp_model = torch.load("models/aesthetic_mlp.pth", map_location=torch.device("cpu"), weights_only=False)
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




::: {.cell .markdown} 


#### CPU execution provider

First, for reference, we'll run the MLP ONNX model with the `CPUExecutionProvider`:

:::




::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
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
onnx_model_path = "models/aesthetic_mlp.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```
:::

<!-- placeholder: update with real benchmark numbers -->


::: {.cell .markdown} 

#### TensorRT execution provider


The TensorRT execution provider will optimize the model for inference on NVIDIA GPUs. It will take a long time to run this cell, because it spends a lot of time optimizing the model (finding the best subgraphs, etc.) - but once the model is loaded, its inference time will be much faster than any of our previous tests.


:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['TensorrtExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```
:::

<!-- placeholder: update with real benchmark numbers -->


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
    -e AESTHETIC_DATA_DIR=/mnt/aesthetic-hub \
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
onnx_model_path = "models/aesthetic_mlp.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['OpenVINOExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```
:::


<!-- placeholder: update with real benchmark numbers -->

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::
