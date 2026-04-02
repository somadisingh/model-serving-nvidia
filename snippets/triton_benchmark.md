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
cp ~/model-serving-nvidia/workspace/models/flickr_global.onnx ~/model-serving-nvidia/models_triton/flickr_global/1/model.onnx
cp ~/model-serving-nvidia/workspace/models/flickr_personalized.onnx ~/model-serving-nvidia/models_triton/flickr_personalized/1/model.onnx
```

3. Bring up the Triton containers:

```bash
# runs on node-serve-model
docker compose -f ~/model-serving-nvidia/docker/docker-compose-triton.yaml up -d
```

4. Verify the server is ready:

```bash
# runs on node-serve-model
docker logs triton_server 2>&1 | tail -5
```

You should see `Started GRPCInferenceService` and `Started HTTPService`.

5. Get the Jupyter token:

```bash
# runs on node-serve-model
docker exec jupyter_triton jupyter server list
```

6. Open this notebook at `http://<FLOATING_IP>:8888`.

:::

::: {.cell .code}
```python
import numpy as np
import time
import tritonclient.http as httpclient
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

for _ in range(num_trials):
    start = time.time()
    infer_global(client, single_emb)
    latencies.append(time.time() - start)

latencies = np.array(latencies)
print(f"Global MLP — Triton Single Sample (n={num_trials})")
print(f"  Median latency:       {np.median(latencies)*1000:.2f} ms")
print(f"  95th percentile:      {np.percentile(latencies, 95)*1000:.2f} ms")
print(f"  99th percentile:      {np.percentile(latencies, 99)*1000:.2f} ms")
print(f"  Throughput:           {num_trials / latencies.sum():.2f} infer/s")
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

Triton ships `perf_analyzer` inside its container. We can run it from the host or install it in another container. For convenience, we'll run it from the Jupyter container.

Note: `perf_analyzer` generates synthetic input data matching the model's input shape.

### Install perf_analyzer

If `perf_analyzer` is not already available in your Jupyter container, run it from the Triton container instead:

```bash
# runs on node-serve-model
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --concurrency-range 1
```

### Concurrency sweep — Global MLP

Run these commands from the **host** (or from inside the Triton container):

```bash
# Concurrency = 1 (baseline)
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 1

# Concurrency = 8
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 8

# Concurrency = 16
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 16
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
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1  --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 8  --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 16 --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 32 --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 64 --shape input:768 --concurrency-range 1
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
nano ~/model-serving-nvidia/models_triton/flickr_global/config.pbtxt
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
docker compose -f ~/model-serving-nvidia/docker/docker-compose-triton.yaml up triton_server --force-recreate -d
```

Wait for it to be ready, then benchmark:
```bash
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --concurrency-range 8
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
nano ~/model-serving-nvidia/models_triton/flickr_global/config.pbtxt
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
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --request-rate-range 200 --request-distribution poisson

# With dynamic batching
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --request-rate-range 200 --request-distribution poisson

# Higher request rate with dynamic batching
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1 --shape input:768 --request-rate-range 500 --request-distribution poisson
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
docker compose -f ~/model-serving-nvidia/docker/docker-compose-triton.yaml down
```

:::
