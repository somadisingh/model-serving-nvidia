::: {.cell .markdown}

# FastAPI Serving Benchmark

This notebook benchmarks the aesthetic scoring MLP served via a **FastAPI + ONNX Runtime** endpoint.

The FastAPI server exposes four endpoints:
- `POST /predict/global` — single global MLP prediction
- `POST /predict/global/batch` — batch global MLP prediction
- `POST /predict/personalized` — single personalized MLP prediction
- `POST /predict/personalized/batch` — batch personalized MLP prediction

Each endpoint accepts **pre-computed CLIP ViT-L/14 embeddings** (768-dim float32 vectors).

## Prerequisites

Before running this notebook:

1. Make sure you have generated the ONNX models by running notebooks 6 and 7 (at minimum `6_measure_onnx.ipynb` to produce `models/flickr_global.onnx` and `models/flickr_personalized.onnx`).

2. Stop the current Jupyter GPU container to free port 8888:

```bash
# runs on node-serve-model
docker stop jupyter
```

3. On the Chameleon host, bring up the FastAPI containers:

```bash
# runs on node-serve-model
docker compose -f ~/aesthetic-hub-serving/docker/docker-compose-fastapi.yaml up -d
```

4. To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). Run

```bash
# runs on node-serve-model
docker exec jupyter_fastapi jupyter server list
```

and look for a line like

```
http://localhost:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface that is running *on your compute instance*.

Then, in the file browser on the left side, open the "work" directory and then click on the `9_fastapi_benchmark.ipynb` notebook to continue.

:::

::: {.cell .code}
```python
import requests
import time
import numpy as np
import concurrent.futures
```
:::

::: {.cell .markdown}

## Resource monitoring

The `ResourceMonitor` class polls `psutil` (CPU and RAM) in a background thread during benchmarks. Because the Jupyter container in this setup does not have direct GPU access, **server-side GPU metrics** are best observed on the host with:

```bash
# runs on node-serve-model (in a separate terminal)
watch -n 1 nvidia-smi
# or
docker stats fastapi_server
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

## Health check

Verify the FastAPI server is reachable.

:::

::: {.cell .code}
```python
FASTAPI_URL = "http://fastapi_server:8000"

resp = requests.get(f"{FASTAPI_URL}/health")
print(resp.status_code, resp.json())
```
:::

::: {.cell .markdown}

## Prepare test embeddings

We create random 768-dim embeddings (L2-normalized) to simulate CLIP outputs. The MLP model doesn't care about semantic content for latency benchmarking.

:::

::: {.cell .code}
```python
# Generate a single random 768-dim embedding (normalized)
rng = np.random.default_rng(42)
single_emb = rng.standard_normal(768).astype(np.float32)
single_emb = single_emb / np.linalg.norm(single_emb)

# Generate a batch of 32 embeddings
batch_emb = rng.standard_normal((32, 768)).astype(np.float32)
batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)

print(f"Single embedding shape: {single_emb.shape}")
print(f"Batch embeddings shape: {batch_emb.shape}")
```
:::

::: {.cell .markdown}

---

## Part 1: Global MLP — Single Request Latency

Send sequential requests one at a time and measure round-trip latency.

:::

::: {.cell .code}
```python
url = f"{FASTAPI_URL}/predict/global"
payload = {"embedding": single_emb.tolist()}

# Quick sanity check
resp = requests.post(url, json=payload)
print(f"Status: {resp.status_code}, Response: {resp.json()}")
```
:::

::: {.cell .code}
```python
num_requests = 200
latencies = []

for _ in range(num_requests):
    start = time.time()
    resp = requests.post(url, json=payload)
    latencies.append(time.time() - start)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code}")

latencies = np.array(latencies)
print(f"Global MLP — Sequential Single Requests (n={num_requests})")
print(f"  Median latency:       {np.median(latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(latencies, 95)*1000:.2f} ms")
print(f"  99th percentile:      {np.percentile(latencies, 99)*1000:.2f} ms")
print(f"  Throughput:           {num_requests / latencies.sum():.2f} req/s")
```
:::

::: {.cell .markdown}

## Part 2: Global MLP — Batch Request Latency

Send a batch of 32 embeddings in a single request.

:::

::: {.cell .code}
```python
batch_url = f"{FASTAPI_URL}/predict/global/batch"
batch_payload = {"embeddings": batch_emb.tolist()}

num_requests = 200
batch_latencies = []

for _ in range(num_requests):
    start = time.time()
    resp = requests.post(batch_url, json=batch_payload)
    batch_latencies.append(time.time() - start)

batch_latencies = np.array(batch_latencies)
batch_throughput = (num_requests * 32) / batch_latencies.sum()
print(f"Global MLP — Sequential Batch Requests (batch_size=32, n={num_requests})")
print(f"  Median latency:       {np.median(batch_latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(batch_latencies, 95)*1000:.2f} ms")
print(f"  Throughput:           {batch_throughput:.2f} samples/s")
```
:::

::: {.cell .markdown}

## Part 3: Global MLP — Concurrent Requests

Simulate multiple clients sending concurrent single requests. This tests queuing behavior.

:::

::: {.cell .code}
```python
def send_single_request(payload):
    start = time.time()
    resp = requests.post(f"{FASTAPI_URL}/predict/global", json=payload)
    elapsed = time.time() - start
    if resp.status_code == 200:
        return elapsed
    return None

def run_concurrent_test(num_requests, payload, max_workers):
    times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_single_request, payload) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            if result is not None:
                times.append(result)
    return np.array(times)
```
:::

::: {.cell .code}
```python
for concurrency in [1, 4, 8, 16]:
    num_requests = 500
    monitor.start()
    wall_start = time.time()
    times = run_concurrent_test(num_requests, payload, max_workers=concurrency)
    wall_time = time.time() - wall_start
    monitor.stop()
    throughput = num_requests / wall_time
    
    print(f"\nConcurrency={concurrency} (n={num_requests})")
    print(f"  Median latency:       {np.median(times)*1000:.2f} ms")
    print(f"  95th percentile:      {np.percentile(times, 95)*1000:.2f} ms")
    print(f"  99th percentile:      {np.percentile(times, 99)*1000:.2f} ms")
    print(f"  Throughput:           {throughput:.2f} req/s")
    monitor.summary(f"Global concurrent={concurrency}")
```
:::

::: {.cell .markdown}

---

## Part 4: Personalized MLP

Repeat the same measurements for the personalized model endpoint, which takes an additional `user_idx` input.

:::

::: {.cell .code}
```python
personal_url = f"{FASTAPI_URL}/predict/personalized"
personal_payload = {"embedding": single_emb.tolist(), "user_idx": 0}

# Sanity check
resp = requests.post(personal_url, json=personal_payload)
print(f"Status: {resp.status_code}, Response: {resp.json()}")
```
:::

::: {.cell .code}
```python
# Sequential single request latency
num_requests = 200
personal_latencies = []

for _ in range(num_requests):
    start = time.time()
    resp = requests.post(personal_url, json=personal_payload)
    personal_latencies.append(time.time() - start)

personal_latencies = np.array(personal_latencies)
print(f"Personalized MLP — Sequential Single Requests (n={num_requests})")
print(f"  Median latency:       {np.median(personal_latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(personal_latencies, 95)*1000:.2f} ms")
print(f"  99th percentile:      {np.percentile(personal_latencies, 99)*1000:.2f} ms")
print(f"  Throughput:           {num_requests / personal_latencies.sum():.2f} req/s")
```
:::

::: {.cell .code}
```python
# Batch request
personal_batch_url = f"{FASTAPI_URL}/predict/personalized/batch"
personal_batch_payload = {
    "embeddings": batch_emb.tolist(),
    "user_indices": [0] * 32
}

num_requests = 200
personal_batch_latencies = []

for _ in range(num_requests):
    start = time.time()
    resp = requests.post(personal_batch_url, json=personal_batch_payload)
    personal_batch_latencies.append(time.time() - start)

personal_batch_latencies = np.array(personal_batch_latencies)
pb_throughput = (num_requests * 32) / personal_batch_latencies.sum()
print(f"Personalized MLP — Sequential Batch Requests (batch_size=32, n={num_requests})")
print(f"  Median latency:       {np.median(personal_batch_latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(personal_batch_latencies, 95)*1000:.2f} ms")
print(f"  Throughput:           {pb_throughput:.2f} samples/s")
```
:::

::: {.cell .code}
```python
# Concurrent single requests for personalized model
def send_personal_request(payload):
    start = time.time()
    resp = requests.post(f"{FASTAPI_URL}/predict/personalized", json=payload)
    elapsed = time.time() - start
    if resp.status_code == 200:
        return elapsed
    return None

for concurrency in [1, 4, 8, 16]:
    num_requests = 500
    wall_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_personal_request, personal_payload) for _ in range(num_requests)]
        times = [f.result() for f in concurrent.futures.as_completed(futures) if f.result() is not None]
    wall_time = time.time() - wall_start
    times = np.array(times)
    throughput = num_requests / wall_time
    
    print(f"\nPersonalized Concurrency={concurrency} (n={num_requests})")
    print(f"  Median latency:       {np.median(times)*1000:.2f} ms")
    print(f"  95th percentile:      {np.percentile(times, 95)*1000:.2f} ms")
    print(f"  Throughput:           {throughput:.2f} req/s")
```
:::

::: {.cell .markdown}

---

## Summary

After running all cells above, fill in the results:

:::

::: {.cell .code}
```python
print("FastAPI Serving Benchmark Summary")
print("=" * 60)
print(f"{'Scenario':<45} {'Median (ms)':>10} {'p95 (ms)':>10}")
print("-" * 65)
print(f"{'Global single (sequential)':<45} {np.median(latencies)*1000:>10.2f} {np.percentile(latencies, 95)*1000:>10.2f}")
print(f"{'Global batch=32 (sequential)':<45} {np.median(batch_latencies)*1000:>10.2f} {np.percentile(batch_latencies, 95)*1000:>10.2f}")
print(f"{'Personalized single (sequential)':<45} {np.median(personal_latencies)*1000:>10.2f} {np.percentile(personal_latencies, 95)*1000:>10.2f}")
print(f"{'Personalized batch=32 (sequential)':<45} {np.median(personal_batch_latencies)*1000:>10.2f} {np.percentile(personal_batch_latencies, 95)*1000:>10.2f}")
```
:::

::: {.cell .markdown}

When you are done, download the fully executed notebook for later reference.

Then, bring down the FastAPI service:

```bash
# runs on node-serve-model
docker compose -f ~/aesthetic-hub-serving/docker/docker-compose-fastapi.yaml down
```

:::
