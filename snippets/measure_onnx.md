

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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from PIL import Image
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load CLIP model for computing image embeddings
device = torch.device("cpu")  # ONNX sessions use CPUExecutionProvider
clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ViT encoding uses GPU
clip_model, clip_preprocess = clip.load("ViT-L/14", device=clip_device)

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

First, let's load our saved PyTorch model, and convert it to ONNX using PyTorch's built-in `torch.onnx.export`:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_path = "models/inference_only/flickr_global_best_inference_only.pth"  
device = torch.device("cpu")
model = GlobalMLP()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()

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
    image_features = clip_model.encode_image(images.to(clip_device))
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
    sample_features = clip_model.encode_image(sample_image[:1].to(clip_device))
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
    batch_features = clip_model.encode_image(batch_images.to(clip_device))
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

::: {.cell .markdown}

#### Quality metrics — Global FP32 ONNX baseline

These metrics show how well the **FP32 ONNX model** predicts aesthetic scores on the held-out test split. Establish this baseline before quantizing in notebook 7.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Quality metrics: full test split — Global FP32 ONNX baseline
global_manifest_q = pd.read_csv(os.path.join(data_dir, "splits", "flickr_global_manifest.csv"))
test_g_q = global_manifest_q[global_manifest_q["split"] == "inference"].reset_index(drop=True)
image_root_q = os.path.join(data_dir, "40K")
print(f"Test set: {len(test_g_q)} images — running CLIP encoding + ONNX inference...")
print("(GPU CLIP encoding — should complete in 1-3 minutes.)")

all_preds_g, all_targets_g = [], []
with torch.no_grad():
    for _i in range(0, len(test_g_q), 32):
        _batch = test_g_q.iloc[_i:_i+32]
        _imgs, _tgts = [], []
        for _, _row in _batch.iterrows():
            try:
                _imgs.append(clip_preprocess(Image.open(os.path.join(image_root_q, _row["image_name"])).convert("RGB")))
                _tgts.append(_row["global_score"])
            except Exception:
                pass
        if not _imgs:
            continue
        _feats = clip_model.encode_image(torch.stack(_imgs).to(clip_device))
        _embs = normalized(_feats.cpu().numpy()).astype(np.float32)
        _preds = ort_session.run(None, {ort_session.get_inputs()[0].name: _embs})[0].flatten()
        all_preds_g.extend(_preds.tolist())
        all_targets_g.extend(_tgts)
        if (_i // 32) % 20 == 0:
            print(f"  {min(_i + 32, len(test_g_q))}/{len(test_g_q)} images ...")

all_preds_g  = np.array(all_preds_g,  dtype=np.float32)
all_targets_g = np.array(all_targets_g, dtype=np.float32)
mae_g   = np.mean(np.abs(all_preds_g - all_targets_g))
rmse_g  = np.sqrt(np.mean((all_preds_g - all_targets_g) ** 2))
plcc_g, _ = pearsonr(all_preds_g, all_targets_g)
srcc_g, _ = spearmanr(all_preds_g, all_targets_g)
bin_acc_g = np.mean((all_preds_g >= 0.5) == (all_targets_g >= 0.5))
auc_g   = roc_auc_score((all_targets_g >= 0.5).astype(int), all_preds_g)
print(f"\n{'─'*55}")
print(f"Quality metrics — Global FP32 ONNX (baseline)")
print(f"{'─'*55}")
print(f"  N:                {len(all_preds_g)}")
print(f"  MAE:              {mae_g:.4f}")
print(f"  RMSE:             {rmse_g:.4f}")
print(f"  PLCC:             {plcc_g:.4f}")
print(f"  SRCC:             {srcc_g:.4f}")
print(f"  Binary accuracy:  {bin_acc_g:.4f}  (threshold=0.5)")
print(f"  AUC-ROC:          {auc_g:.4f}")
print("Compare with quantized / optimized variants in notebook 7.")
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
personal_model_path = "models/inference_only/flickr_personalized_best_inference_only.pth"
_p_state = torch.load(personal_model_path, map_location=device, weights_only=False)
_num_users = _p_state["user_embedding.weight"].shape[0]
personal_model = PersonalizedMLP(num_users=_num_users)
personal_model.load_state_dict(_p_state)
personal_model.eval()

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
    image_features = clip_model.encode_image(images.to(clip_device))
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
    sample_features = clip_model.encode_image(sample_image[:1].to(clip_device))
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
    batch_features = clip_model.encode_image(batch_images.to(clip_device))
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

#### Quality metrics — Personalized FP32 ONNX baseline

Per-user SRCC and MAE across every annotator in the test split.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Quality metrics: Personalized FP32 ONNX — per-user SRCC and MAE
p_manifest_q = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
test_p_q = p_manifest_q[p_manifest_q["split"] == "inference"].reset_index(drop=True)
image_root_p = os.path.join(data_dir, "40K")
input_names_p = [i.name for i in personal_ort_session.get_inputs()]
test_workers_q = [w for w in test_p_q["worker_id"].unique() if w in user2idx]
print(f"Personalized test: {len(test_p_q)} rows, {len(test_workers_q)} known workers")
print("(Running CLIP + ONNX per worker — GPU CLIP encoding, ~5-10 minutes.)")

per_user_srcc_q, per_user_mae_q = [], []
for worker_id in test_workers_q:
    uid = user2idx[worker_id]
    worker_df = test_p_q[test_p_q["worker_id"] == worker_id].reset_index(drop=True)
    if len(worker_df) < 3:
        continue
    w_preds, w_targets = [], []
    with torch.no_grad():
        for _i in range(0, len(worker_df), 32):
            _batch = worker_df.iloc[_i:_i+32]
            _imgs, _tgts = [], []
            for _, _row in _batch.iterrows():
                try:
                    _imgs.append(clip_preprocess(Image.open(os.path.join(image_root_p, _row["image_name"])).convert("RGB")))
                    _tgts.append(_row["worker_score_norm"])
                except Exception:
                    pass
            if not _imgs:
                continue
            _feats = clip_model.encode_image(torch.stack(_imgs).to(clip_device))
            _embs = normalized(_feats.cpu().numpy()).astype(np.float32)
            _uids = np.full(len(_imgs), uid, dtype=np.int64)
            _preds = personal_ort_session.run(None, {input_names_p[0]: _embs, input_names_p[1]: _uids})[0].flatten()
            w_preds.extend(_preds.tolist())
            w_targets.extend(_tgts)
    if len(w_preds) < 3:
        continue
    w_p = np.array(w_preds)
    w_t = np.array(w_targets)
    per_user_srcc_q.append(spearmanr(w_p, w_t)[0])
    per_user_mae_q.append(np.mean(np.abs(w_p - w_t)))

print(f"\n{'─'*55}")
print(f"Quality metrics — Personalized FP32 ONNX (baseline)")
print(f"{'─'*55}")
print(f"  Users evaluated:    {len(per_user_srcc_q)}")
print(f"  Mean per-user SRCC: {np.mean(per_user_srcc_q):.4f}")
print(f"  Mean per-user MAE:  {np.mean(per_user_mae_q):.4f}")
print("Compare with quantized variants in notebook 7.")
```
:::


::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the `flickr_global.onnx` and `flickr_personalized.onnx` models from inside the `models` directory.

:::
