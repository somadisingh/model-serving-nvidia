

::: {.cell .markdown}

## Measure inference performance of ONNX model on CPU 

To squeeze even more inference performance out of our model, we are going to convert it to ONNX format, which allows models from different frameworks (PyTorch, Tensorflow, Keras), to be deployed on a variety of different hardware platforms (CPU, GPU, edge devices), using many optimizations (graph optimizations, quantization, target device-specific implementations, and more).

After finishing this section, you should know:

* how to convert a PyTorch model to ONNX
* how to measure the inference latency and batch throughput of the ONNX model

and then you will use it to evaluate the optimized models you develop in the next section.

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.

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
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load CLIP model for computing image embeddings
device = torch.device("cpu")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Prepare test dataset using CLIP's preprocessing
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```
:::



::: {.cell .markdown}

First, let's load our saved PyTorch model, and convert it to ONNX using PyTorch's built-in `torch.onnx.export`:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_path = "models/flickr_global_best_inference_only.pth"  
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)

onnx_model_path = "models/flickr_global.onnx"
# MLP expects a 768-dim CLIP embedding as input
dummy_input = torch.randn(1, 768)
torch.onnx.export(model, dummy_input, onnx_model_path,
                  export_params=True, opset_version=20,
                  do_constant_folding=True, input_names=['input'],
                  output_names=['output'], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print(f"ONNX model saved to {onnx_model_path}")

onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
```
:::

::: {.cell .markdown}

## Create an inference session

Now, we can evaluate our model! To use an ONNX model, we create an *inference session*, and then use the model within that session.

For this first ONNX baseline, we will explicitly disable graph optimizations, so that later we can clearly see the effect when we enable them. Let's start an inference session:


:::



::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

ort_session = ort.InferenceSession(
    onnx_model_path,
    sess_options=session_options,
    providers=['CPUExecutionProvider']
)
```
:::



::: {.cell .markdown}

and let's double check the execution provider that will be used in this session:

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
ort_session.get_providers()
```
:::




::: {.cell .markdown}

#### Sample predictions


First, let's verify the model produces reasonable aesthetic scores:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    embeddings = normalized(image_features.cpu().numpy()).astype(np.float32)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: embeddings})[0]
    scores = outputs.flatten()
    mean_score = scores.mean()
    std_score = scores.std()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (0-1):")
for i in range(min(5, len(scores))):
    print(f"  Image {i+1}: {scores[i]:.2f}")
print(f"\nBatch mean: {mean_score:.2f}, std: {std_score:.2f}")
```
:::

::: {.cell .markdown}

#### Model size

We are also concerned with the size of the ONNX model on disk. It will be similar to the equivalent PyTorch model size (to start!)

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::




::: {.cell .markdown}

#### Inference latency

Now, we'll measure how long it takes the model to return a prediction for a single sample. We will run 100 trials, and then compute aggregate statistics.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 100  # Number of trials

# Pre-compute a single CLIP embedding for benchmarking
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    single_embedding = normalized(sample_features.cpu().numpy()).astype(np.float32)

# Warm-up run
ort_session.run(None, {ort_session.get_inputs()[0].name: single_embedding})

latencies = []
for _ in range(num_trials):
    start_time = time.time()
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_embedding})
    latencies.append(time.time() - start_time)
```
:::



::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```
:::

::: {.cell .markdown}

#### Batch throughput 

Finally, we'll measure the rate at which the model can return predictions for batches of data. 

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_batches = 50  # Number of trials

# Pre-compute a batch of CLIP embeddings for benchmarking
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    batch_embeddings = normalized(batch_features.cpu().numpy()).astype(np.float32)

# Warm-up run
ort_session.run(None, {ort_session.get_inputs()[0].name: batch_embeddings})

batch_times = []
for _ in range(num_batches):
    start_time = time.time()
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_embeddings})
    batch_times.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
batch_fps = (batch_embeddings.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::



::: {.cell .markdown}

#### Summary of results

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Mean Predicted Score: {mean_score:.2f} (std: {std_score:.2f})")
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::

<!-- summary for flickr_global

Model Size on Disk: 8.92 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.92 ms
Inference Latency (single sample, 95th percentile): 9.15 ms
Inference Latency (single sample, 99th percentile): 9.41 ms
Inference Throughput (single sample): 112.06 FPS
Batch Throughput: 993.48 FPS

Model Size on Disk: 8.92 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.64 ms
Inference Latency (single sample, 95th percentile): 10.57 ms
Inference Latency (single sample, 99th percentile): 11.72 ms
Inference Latency (single sample, std error): 0.04 ms
Inference Throughput (single sample): 102.52 FPS
Batch Throughput: 1083.57 FPS

Accuracy: 90.59% (3032/3347 correct)
Model Size on Disk: 8.92 MB
Inference Latency (single sample, median): 16.24 ms
Inference Latency (single sample, 95th percentile): 18.06 ms
Inference Latency (single sample, 99th percentile): 18.72 ms
Inference Throughput (single sample): 63.51 FPS
Batch Throughput: 1103.28 FPS


-->


<!-- summary for flickr_global with graph optimization

Model Size on Disk: 8.91 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.31 ms
Inference Latency (single sample, 95th percentile): 9.47 ms
Inference Latency (single sample, 99th percentile): 9.71 ms
Inference Throughput (single sample): 107.22 FPS
Batch Throughput: 1091.58 FPS

Model Size on Disk: 8.91 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.95 ms
Inference Latency (single sample, 95th percentile): 10.14 ms
Inference Latency (single sample, 99th percentile): 10.70 ms
Inference Latency (single sample, std error): 0.02 ms
Inference Throughput (single sample): 100.18 FPS
Batch Throughput: 1022.77 FPS

Model Size on Disk: 8.91 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.55 ms
Inference Latency (single sample, 95th percentile): 10.58 ms
Inference Latency (single sample, 99th percentile): 11.14 ms
Inference Latency (single sample, std error): 0.04 ms
Inference Throughput (single sample): 102.97 FPS
Batch Throughput: 1079.81 FPS


-->


<!-- 

(Intel CPU)

Accuracy: 90.59% (3032/3347 correct)
Model Size on Disk: 8.92 MB
Inference Latency (single sample, median): 4.53 ms
Inference Latency (single sample, 95th percentile): 4.63 ms
Inference Latency (single sample, 99th percentile): 4.99 ms
Inference Throughput (single sample): 218.75 FPS
Batch Throughput: 2519.80 FPS


-->


::: {.cell .markdown}

---

## Personalized MLP: ONNX Conversion and Baseline

The personalized model takes two inputs: a 768-dim CLIP embedding and a user index (integer). The user index is looked up in an `nn.Embedding` table to produce a 64-dim user vector, which is concatenated with the CLIP embedding before passing through the MLP. Let's convert it to ONNX and benchmark it.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load personalized model and get valid user indices
personal_model_path = "models/flickr_personalized_best_inference_only.pth"
personal_model = torch.load(personal_model_path, map_location=device, weights_only=False)

data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
personal_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
seen_workers = sorted(personal_manifest.loc[personal_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique())
user2idx = {u: i for i, u in enumerate(seen_workers)}
num_users = len(user2idx)
print(f"Personalized model: {num_users} known users")
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Export personalized model to ONNX with two inputs
personal_onnx_path = "models/flickr_personalized.onnx"
dummy_embedding = torch.randn(1, 768)
dummy_user_idx = torch.tensor([0], dtype=torch.long)

torch.onnx.export(
    personal_model, 
    (dummy_embedding, dummy_user_idx), 
    personal_onnx_path,
    export_params=True, opset_version=20,
    do_constant_folding=True,
    input_names=['embedding', 'user_idx'],
    output_names=['output'],
    dynamic_axes={
        "embedding": {0: "batch_size"}, 
        "user_idx": {0: "batch_size"}, 
        "output": {0: "batch_size"}
    }
)

print(f"Personalized ONNX model saved to {personal_onnx_path}")
personal_onnx_model = onnx.load(personal_onnx_path)
onnx.checker.check_model(personal_onnx_model)
```
:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Create inference session (no graph optimizations for baseline)
personal_session_options = ort.SessionOptions()
personal_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

personal_ort_session = ort.InferenceSession(
    personal_onnx_path,
    sess_options=personal_session_options,
    providers=['CPUExecutionProvider']
)
print(f"Inputs: {[(i.name, i.shape, i.type) for i in personal_ort_session.get_inputs()]}")
```
:::


::: {.cell .markdown}

#### Sample predictions

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    p_embeddings = normalized(image_features.cpu().numpy()).astype(np.float32)
    p_user_idx = np.zeros(p_embeddings.shape[0], dtype=np.int64)
    
    outputs = personal_ort_session.run(None, {
        'embedding': p_embeddings, 
        'user_idx': p_user_idx
    })[0]
    p_scores = outputs.flatten()
    p_mean_score = p_scores.mean()
    p_std_score = p_scores.std()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (personalized, 0-1):")
for i in range(min(5, len(p_scores))):
    print(f"  Image {i+1}: {p_scores[i]:.2f}")
print(f"\nBatch mean: {p_mean_score:.2f}, std: {p_std_score:.2f}")
```
:::


::: {.cell .markdown}

#### Model size

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
personal_model_size = os.path.getsize(personal_onnx_path) 
print(f"Personalized Model Size on Disk: {personal_model_size / (1e6):.2f} MB")
print(f"Global Model Size on Disk:       {model_size / (1e6):.2f} MB")
```
:::


::: {.cell .markdown}

#### Inference latency

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Pre-compute a single CLIP embedding for benchmarking
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    p_single_embedding = normalized(sample_features.cpu().numpy()).astype(np.float32)
    p_single_user_idx = np.array([0], dtype=np.int64)

# Warm-up run
personal_ort_session.run(None, {'embedding': p_single_embedding, 'user_idx': p_single_user_idx})

p_latencies = []
for _ in range(num_trials):
    start_time = time.time()
    personal_ort_session.run(None, {'embedding': p_single_embedding, 'user_idx': p_single_user_idx})
    p_latencies.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Inference Latency (single sample, median): {np.percentile(p_latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(p_latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(p_latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(p_latencies):.2f} FPS")
```
:::


::: {.cell .markdown}

#### Batch throughput

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_batches = 50

# Pre-compute a batch of CLIP embeddings
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    p_batch_embeddings = normalized(batch_features.cpu().numpy()).astype(np.float32)
    p_batch_user_idx = np.zeros(p_batch_embeddings.shape[0], dtype=np.int64)

# Warm-up run
personal_ort_session.run(None, {'embedding': p_batch_embeddings, 'user_idx': p_batch_user_idx})

p_batch_times = []
for _ in range(num_batches):
    start_time = time.time()
    personal_ort_session.run(None, {'embedding': p_batch_embeddings, 'user_idx': p_batch_user_idx})
    p_batch_times.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
p_batch_fps = (p_batch_embeddings.shape[0] * num_batches) / np.sum(p_batch_times) 
print(f"Personalized Batch Throughput: {p_batch_fps:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Personalized ONNX summary

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Mean Predicted Score: {p_mean_score:.2f} (std: {p_std_score:.2f})")
print(f"Model Size on Disk: {personal_model_size / (1e6):.2f} MB")
print(f"Inference Latency (single sample, median): {np.percentile(p_latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(p_latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(p_latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(p_latencies):.2f} FPS")
print(f"Batch Throughput: {p_batch_fps:.2f} FPS")
```
:::


::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the `flickr_global.onnx` and `flickr_personalized.onnx` models from inside the `models` directory.

:::
