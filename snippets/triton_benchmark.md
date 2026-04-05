::: {.cell .markdown}

# Triton Inference Server Benchmark

This notebook benchmarks the aesthetic scoring MLP head served via **NVIDIA Triton Inference Server** with the ONNX Runtime backend.

Triton serves two models:
- `flickr_global` — Global MLP (input: 768-dim embedding → output: aesthetic score)
- `flickr_personalized` — Personalized MLP (inputs: embedding + user_idx → output: score)

## Prerequisites

Before running this notebook:

1. Generate the ONNX models by running notebook 6 (`6_measure_onnx.ipynb`).

2. Copy models into the Triton model repository:

```bash
# runs on node-serve-model
cp ~/aesthetic-hub-serving/workspace/models/flickr_global.onnx ~/aesthetic-hub-serving/models_triton/flickr_global/1/model.onnx
cp ~/aesthetic-hub-serving/workspace/models/flickr_personalized.onnx ~/aesthetic-hub-serving/models_triton/flickr_personalized/1/model.onnx
```

3. Bring up the Triton containers:

```bash
# runs on node-serve-model
docker compose -f ~/aesthetic-hub-serving/docker/docker-compose-triton.yaml up -d
```

4. Verify the server is ready:

```bash
# runs on node-serve-model
docker logs triton_server 2>&1 | tail -5
```

You should see `Started GRPCInferenceService` and `Started HTTPService`.

5. To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). Run

```bash
# runs on node-serve-model
docker exec jupyter_triton jupyter server list
```

   and look for a line like

       http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

   Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

   Then, in the file browser on the left side, open the "work" directory and then click on the `10_triton_benchmark.ipynb` notebook to continue.

:::

::: {.cell .code}
```python
import numpy as np
import time
import tritonclient.http as httpclient
```
:::

::: {.cell .markdown}

## Resource monitoring

The `ResourceMonitor` class polls `psutil` (CPU and RAM) in a background thread. Because the Jupyter container in this setup does not have direct GPU access, **server-side GPU metrics** can be pulled from Triton's built-in Prometheus metrics endpoint on port 8002, or observed on the host:

```bash
# runs on node-serve-model (in a separate terminal)
watch -n 1 nvidia-smi
# or
docker stats triton_server
```

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

## Server-side GPU metrics via Triton

Triton exposes live GPU and inference statistics in Prometheus format at `http://triton_server:8002/metrics`. The cell below pulls the key GPU metrics — utilization, memory, and per-model inference count — directly from that endpoint.

:::

::: {.cell .code}
```python
import requests as _req

def triton_gpu_stats():
    """Fetch GPU utilization and memory from Triton's Prometheus metrics endpoint."""
    try:
        lines = _req.get("http://triton_server:8002/metrics", timeout=2).text.splitlines()
    except Exception as e:
        print(f"Could not reach Triton metrics endpoint: {e}")
        return
    keys = {
        "nv_gpu_utilization": "GPU utilization (%)",
        "nv_gpu_memory_used_bytes": "GPU memory used (MB)",
        "nv_gpu_memory_total_bytes": "GPU memory total (MB)",
        "nv_inference_count": "Total inferences served",
    }
    print("Triton server-side GPU metrics:")
    for line in lines:
        for key, label in keys.items():
            if line.startswith(key) and not line.startswith("#"):
                val = float(line.split()[-1])
                if "bytes" in key:
                    val /= 1024 ** 2  # bytes -> MB
                print(f"  {label}: {val:.1f}")

triton_gpu_stats()
```
:::

::: {.cell .markdown}

---

## Part 1: Verify Triton is ready and models are loaded

:::

::: {.cell .code}
```python
TRITON_URL = "triton_server:8000"
client = httpclient.InferenceServerClient(url=TRITON_URL)

print(f"Server live: {client.is_server_live()}")
print(f"Server ready: {client.is_server_ready()}")
print()

for model_name in ["flickr_global", "flickr_personalized"]:
    ready = client.is_model_ready(model_name)
    meta = client.get_model_metadata(model_name)
    print(f"Model '{model_name}': ready={ready}")
    for inp in meta['inputs']:
        print(f"  Input:  {inp['name']:>15s}  shape={inp['shape']}  dtype={inp['datatype']}")
    for out in meta['outputs']:
        print(f"  Output: {out['name']:>15s}  shape={out['shape']}  dtype={out['datatype']}")
    print()
```
:::

::: {.cell .markdown}

## Prepare test data

:::

::: {.cell .code}
```python
rng = np.random.default_rng(42)

# Single sample
single_emb = rng.standard_normal(768).astype(np.float32)
single_emb = single_emb / np.linalg.norm(single_emb)
single_emb = single_emb.reshape(1, 768)

# Batch of 32
batch_emb = rng.standard_normal((32, 768)).astype(np.float32)
batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)

# User indices
single_user_idx = np.array([[0]], dtype=np.int64)
batch_user_idx = np.zeros((32, 1), dtype=np.int64)

print(f"Single embedding: {single_emb.shape}")
print(f"Batch embeddings: {batch_emb.shape}")
```
:::

::: {.cell .markdown}

---

## Part 2: Global MLP — Triton Client Benchmark

### Sanity check

:::

::: {.cell .code}
```python
def infer_global(client, embeddings):
    """Send a global model inference request."""
    inputs = [httpclient.InferInput("input", embeddings.shape, "FP32")]
    inputs[0].set_data_from_numpy(embeddings)
    outputs = [httpclient.InferRequestedOutput("output")]
    result = client.infer(model_name="flickr_global", inputs=inputs, outputs=outputs)
    return result.as_numpy("output")

# Sanity check
scores = infer_global(client, single_emb)
print(f"Single prediction: {scores.flatten()[0]:.4f}")

scores = infer_global(client, batch_emb)
print(f"Batch predictions (first 5): {', '.join(f'{s:.4f}' for s in scores.flatten()[:5])}")
```
:::

::: {.cell .markdown}

### Sequential single-sample latency

:::

::: {.cell .code}
```python
num_trials = 500
latencies = []

# Warm-up
for _ in range(20):
    infer_global(client, single_emb)

monitor.start()
for _ in range(num_trials):
    start = time.time()
    infer_global(client, single_emb)
    latencies.append(time.time() - start)
monitor.stop()

latencies = np.array(latencies)
print(f"Global MLP — Triton Single Sample (n={num_trials})")
print(f"  Median latency:       {np.median(latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(latencies, 95)*1000:.2f} ms")
print(f"  99th percentile:      {np.percentile(latencies, 99)*1000:.2f} ms")
print(f"  Throughput:           {num_trials / latencies.sum():.2f} infer/s")
monitor.summary("Global Triton single sample (client-side CPU)")
triton_gpu_stats()
```
:::

::: {.cell .markdown}

### Batch throughput (batch_size=32)

:::

::: {.cell .code}
```python
num_trials = 200
batch_latencies = []

for _ in range(20):
    infer_global(client, batch_emb)

for _ in range(num_trials):
    start = time.time()
    infer_global(client, batch_emb)
    batch_latencies.append(time.time() - start)

batch_latencies = np.array(batch_latencies)
throughput = (num_trials * 32) / batch_latencies.sum()
print(f"Global MLP — Triton Batch=32 (n={num_trials})")
print(f"  Median latency:       {np.median(batch_latencies)*1000:.2f} ms")
print(f"  Throughput:           {throughput:.2f} samples/s")
```
:::

::: {.cell .markdown}

---

## Part 3: Personalized MLP — Triton Client Benchmark

### Sanity check

:::

::: {.cell .code}
```python
def infer_personalized(client, embeddings, user_indices):
    """Send a personalized model inference request."""
    inp_emb = httpclient.InferInput("embedding", embeddings.shape, "FP32")
    inp_emb.set_data_from_numpy(embeddings)
    inp_idx = httpclient.InferInput("user_idx", user_indices.shape, "INT64")
    inp_idx.set_data_from_numpy(user_indices)
    outputs = [httpclient.InferRequestedOutput("output")]
    result = client.infer(model_name="flickr_personalized", inputs=[inp_emb, inp_idx], outputs=outputs)
    return result.as_numpy("output")

scores = infer_personalized(client, single_emb, single_user_idx)
print(f"Single prediction: {scores.flatten()[0]:.4f}")

scores = infer_personalized(client, batch_emb, batch_user_idx)
print(f"Batch predictions (first 5): {', '.join(f'{s:.4f}' for s in scores.flatten()[:5])}")
```
:::

::: {.cell .code}
```python
# Sequential single-sample latency
num_trials = 500
personal_latencies = []

for _ in range(20):
    infer_personalized(client, single_emb, single_user_idx)

for _ in range(num_trials):
    start = time.time()
    infer_personalized(client, single_emb, single_user_idx)
    personal_latencies.append(time.time() - start)

personal_latencies = np.array(personal_latencies)
print(f"Personalized MLP — Triton Single Sample (n={num_trials})")
print(f"  Median latency:       {np.median(personal_latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(personal_latencies, 95)*1000:.2f} ms")
print(f"  99th percentile:      {np.percentile(personal_latencies, 99)*1000:.2f} ms")
print(f"  Throughput:           {num_trials / personal_latencies.sum():.2f} infer/s")
```
:::

::: {.cell .code}
```python
# Batch throughput (batch_size=32)
num_trials = 200
personal_batch_latencies = []

for _ in range(20):
    infer_personalized(client, batch_emb, batch_user_idx)

for _ in range(num_trials):
    start = time.time()
    infer_personalized(client, batch_emb, batch_user_idx)
    personal_batch_latencies.append(time.time() - start)

personal_batch_latencies = np.array(personal_batch_latencies)
pb_throughput = (num_trials * 32) / personal_batch_latencies.sum()
print(f"Personalized MLP — Triton Batch=32 (n={num_trials})")
print(f"  Median latency:       {np.median(personal_batch_latencies)*1000:.2f} ms")
print(f"  Throughput:           {pb_throughput:.2f} samples/s")
```
:::

::: {.cell .markdown}

---

## Part 4: `perf_analyzer` Benchmarks

`perf_analyzer` is Triton's official load-testing CLI. It lives in the **SDK image** (`tritonserver:24.10-py3-sdk`), not the server image, so the compose file includes a lightweight `triton_sdk` sidecar container that shares the server's network namespace.

The tool generates synthetic input tensors matching the model's input shape, fires them at the server, and reports precise latency breakdowns (queue time, compute time) and throughput.

### Concurrency sweep — Global MLP

Run these commands on the **host** (SSH into `node-serve-model`):

```bash
# Concurrency = 1 (baseline)
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 1

# Concurrency = 8
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 8

# Concurrency = 16
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 16
```

Record the **average request latency** and its breakdown:
- `queue` — queuing delay
- `compute infer` — actual inference time
- `throughput` — inferences per second

:::

::: {.cell .code}
```python
# Placeholder: paste perf_analyzer results here for reference
# Concurrency=1:  Avg latency = ____ usec (queue=____, compute infer=____), throughput=____ infer/sec
# Concurrency=8:  Avg latency = ____ usec (queue=____, compute infer=____), throughput=____ infer/sec
# Concurrency=16: Avg latency = ____ usec (queue=____, compute infer=____), throughput=____ infer/sec
```
:::

::: {.cell .markdown}

### Batch-size sweep

Test different batch sizes with a single concurrent client:

```bash
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1  --shape input:768 --concurrency-range 1 --measurement-interval 10000
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 8  --shape input:768 --concurrency-range 1 --measurement-interval 10000
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 16 --shape input:768 --concurrency-range 1 --measurement-interval 10000
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 32 --shape input:768 --concurrency-range 1 --measurement-interval 10000
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 64 --shape input:768 --concurrency-range 1 --measurement-interval 10000
```

:::

::: {.cell .code}
```python
# Placeholder: paste batch-size sweep results
# b=1:  throughput=____ infer/sec, latency=____ usec
# b=8:  throughput=____ infer/sec, latency=____ usec
# b=16: throughput=____ infer/sec, latency=____ usec
# b=32: throughput=____ infer/sec, latency=____ usec
# b=64: throughput=____ infer/sec, latency=____ usec
```
:::

::: {.cell .markdown}

---

## Part 5: Scaling — Multiple Model Instances

By default, we configured 1 instance on GPU 0. Let's test with more instances.

### Scale to 2 instances on GPU 0

On the host, edit the config:

```bash
# runs on node-serve-model
nano ~/aesthetic-hub-serving/models_triton/flickr_global/config.pbtxt
```

Change:
```
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```
to:
```
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]
```

Restart Triton:
```bash
docker compose -f ~/aesthetic-hub-serving/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

Wait for it to be ready, then benchmark:
```bash
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 8
```

Compare queue delay and throughput vs the single-instance case.

:::

::: {.cell .code}
```python
# Placeholder: paste scaling results
# 1 instance, concurrency=8:  throughput=____, queue=____ usec
# 2 instances, concurrency=8: throughput=____, queue=____ usec
```
:::

::: {.cell .markdown}

---

## Part 6: Dynamic Batching

Dynamic batching lets Triton combine multiple individual requests into a batch automatically, absorbing bursts without overprovisioning.

### Enable dynamic batching

Reset to 1 instance, then edit `config.pbtxt`:

```bash
nano ~/aesthetic-hub-serving/models_triton/flickr_global/config.pbtxt
```

Add at the end:
```
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}
```

Restart Triton, then test with Poisson arrivals at various request rates:

```bash
# Without dynamic batching (comment it out first for comparison)
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --request-rate-range 200 --request-distribution poisson

# With dynamic batching
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --request-rate-range 200 --request-distribution poisson

# Higher request rate with dynamic batching
docker exec triton_sdk perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --request-rate-range 500 --request-distribution poisson
```

Check batch statistics:
```bash
curl -s http://localhost:8000/v2/models/flickr_global/versions/1/stats | python3 -m json.tool
```

:::

::: {.cell .code}
```python
# Placeholder: paste dynamic batching results
# Rate=200 (no batching):  avg latency=____ usec, queue=____ usec
# Rate=200 (batching):     avg latency=____ usec, queue=____ usec
# Rate=500 (batching):     avg latency=____ usec, queue=____ usec
```
:::

::: {.cell .markdown}

---

## Summary

:::

::: {.cell .code}
```python
print("Triton Serving Benchmark Summary (Python client)")
print("=" * 60)
print(f"{'Scenario':<45} {'Median (ms)':>10} {'p95 (ms)':>10}")
print("-" * 65)
print(f"{'Global single':<45} {np.median(latencies)*1000:>10.2f} {np.percentile(latencies, 95)*1000:>10.2f}")
print(f"{'Global batch=32':<45} {np.median(batch_latencies)*1000:>10.2f} {np.percentile(batch_latencies, 95)*1000:>10.2f}")
print(f"{'Personalized single':<45} {np.median(personal_latencies)*1000:>10.2f} {np.percentile(personal_latencies, 95)*1000:>10.2f}")
print(f"{'Personalized batch=32':<45} {np.median(personal_batch_latencies)*1000:>10.2f} {np.percentile(personal_batch_latencies, 95)*1000:>10.2f}")
```
:::

::: {.cell .markdown}

When you are done, download the fully executed notebook for later reference.

Then, bring down the Triton service:

```bash
# runs on node-serve-model
docker compose -f ~/aesthetic-hub-serving/docker/docker-compose-triton.yaml down
```

:::
