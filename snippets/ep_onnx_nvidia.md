

::: {.cell .markdown}

### Try a different execution provider

Once a model is in ONNX format, we can use it with many *execution providers*. In ONNX, an execution provider an interface that lets ONNX models run with special hardware-specific capabilities. Until now, we have been using the `CPUExecutionProvider`, but if we use hardware-specific capabilities, e.g. switch out generic implementations of graph operations for implementations that are optimized for specific hardware, we can execute exactly the same model, much faster.

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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Prepare test dataset
data_dir = os.getenv("AESTHETIC_DATA_DIR", "aesthetic-hub")
val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```
:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
def benchmark_session(ort_session):

    print(f"Execution provider: {ort_session.get_providers()}")

    ## Benchmark accuracy

    correct = 0
    total = 0
    for images, labels in test_loader:
        images_np = images.numpy()
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: images_np})[0]
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    accuracy = (correct / total) * 100

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials

    # Get a single sample from the test data

    single_sample, _ = next(iter(test_loader))  
    single_sample = single_sample[:1].numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput

    num_batches = 50  # Number of trials

    # Get a batch from the test data
    batch_input, _ = next(iter(test_loader))  
    batch_input = batch_input.numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")

```
:::




::: {.cell .markdown} 


#### CPU execution provider

First, for reference, we'll repeat our performance test for the (unquantized model with) `CPUExecutionProvider`:

:::




::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```
:::

<!--
Execution provider: ['CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 9.93 ms
Inference Latency (single sample, 95th percentile): 14.20 ms
Inference Latency (single sample, 99th percentile): 14.43 ms
Inference Throughput (single sample): 91.10 FPS
Batch Throughput: 1042.47 FPS
-->


::: {.cell .markdown} 

#### CUDA execution provider


Before we can use CUDA and TensorRT execution providers, we need to switch from the `jupyter-onnx-base` image to the `jupyter-onnx-gpu` image.

Close this Jupyter server tab - you will reopen it shortly, with a new token.

Go back to your SSH session on "node-serve-model", and stop the current Jupyter server with:

```bash
# runs on node-serve-model
docker stop jupyter
```

Build the GPU image:

```bash
# runs on node-serve-model
docker build -t jupyter-onnx-gpu -f serve-model-chi/docker/Dockerfile.jupyter-onnx-nvidia .
```

Then launch a new one with the GPU image:

```bash
# runs on node-serve-model
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/serve-model-chi/workspace:/home/jovyan/work/ \
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

Run the three cells at the top, which `import` libraries, set up the data loaders, and define the `benchmark_session` function. Then continue with CUDA and TensorRT:


Next, we'll try it with the CUDA execution provider, which will execute the model on the GPU:

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

<!--
Execution provider: ['CUDAExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 0.89 ms
Inference Latency (single sample, 95th percentile): 0.90 ms
Inference Latency (single sample, 99th percentile): 0.91 ms
Inference Throughput (single sample): 1117.06 FPS
Batch Throughput: 5181.99 FPS
-->


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

<!--
Execution provider: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 0.63 ms
Inference Latency (single sample, 95th percentile): 0.64 ms
Inference Latency (single sample, 99th percentile): 0.70 ms
Inference Throughput (single sample): 1572.61 FPS
Batch Throughput: 9274.45 FPS
-->


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
docker build -t jupyter-onnx-openvino -f serve-model-chi/docker/Dockerfile.jupyter-onnx-openvino .
```

Then, launch a container with the OpenVINO image:

```bash
# runs on node-serve-model
docker run  -d --rm  -p 8888:8888 \
    --shm-size 16G \
    -v ~/serve-model-chi/workspace:/home/jovyan/work/ \
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

Run the three cells at the top, which `import` libraries, set up the data loaders, and define the `benchmark_session` function. Then, skip to the OpenVINO section and run:

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


<!--

On AMD EPYC

Execution provider: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 1.39 ms
Inference Latency (single sample, 95th percentile): 1.89 ms
Inference Latency (single sample, 99th percentile): 1.92 ms
Inference Throughput (single sample): 646.63 FPS
Batch Throughput: 1624.30 FPS

On Intel

Execution provider: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 1.55 ms
Inference Latency (single sample, 95th percentile): 1.76 ms
Inference Latency (single sample, 99th percentile): 1.81 ms
Inference Throughput (single sample): 663.72 FPS
Batch Throughput: 2453.48 FPS

-->

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::
