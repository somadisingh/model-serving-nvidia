
::: {.cell .markdown}

## Measure inference performance of PyTorch model on CPU 

First, we are going to measure the inference performance of an already-trained PyTorch model on CPU. After completing this section, you should understand:

* how to measure the inference latency of a PyTorch model
* how to measure the throughput of batch inference of a PyTorch model
* how to compare eager model execution vs a compiled model

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
```
:::


::: {.cell .markdown}

First, let's load our saved model in evaluation mode. Note that for now, we will use the CPU for inference, not GPU.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_path = "models/aesthetic_mlp.pth"  
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()  
```
:::

::: {.cell .markdown}

and also prepare our test dataset:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
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


::: {.cell .markdown}

We will measure:

* the size of the model on disk
* the latency when doing inference on single samples
* the throughput when doing inference on batches of data
* and the test accuracy

:::


::: {.cell .markdown}

#### Model size

We'll start with model size. Our `aesthetic_mlp.pth` is a lightweight MLP head (768 → 1024 → 128 → 64 → 16 → 1) that maps CLIP ViT-L/14 embeddings to aesthetic scores, so it is very small.
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_size = os.path.getsize(model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::

::: {.cell .markdown}

#### Test accuracy

Next, we'll measure the accuracy of this model on the test data

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get predicted class index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
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

# Get a single sample from the test data

single_sample, _ = next(iter(test_loader))  
single_sample = single_sample[0].unsqueeze(0)  

# Warm-up run 
with torch.no_grad():
    model(single_sample)

latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(single_sample)
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

# Get a batch from the test data
batch_input, _ = next(iter(test_loader))  

# Warm-up run 
with torch.no_grad():
    model(batch_input)

batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = model(batch_input)
        batch_times.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Summary of results

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::

<!-- 

compute_gigaio 

  Model name:             AMD EPYC 7763 64-Core Processor
    CPU family:           25
    Model:                1
    Thread(s) per core:   2
    Core(s) per socket:   64

-->


<!-- summary for mobilenet model

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 60.16 ms
Inference Latency (single sample, 95th percentile): 77.22 ms
Inference Latency (single sample, 99th percentile): 77.37 ms
Inference Throughput (single sample): 15.82 FPS
Batch Throughput: 83.66 FPS


Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 73.97 ms
Inference Latency (single sample, 95th percentile): 83.16 ms
Inference Latency (single sample, 99th percentile): 83.94 ms
Inference Throughput (single sample): 13.34 FPS
Batch Throughput: 98.80 FPS

-->


<!-- summary for mobilenet compiled model

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 26.92 ms
Inference Latency (single sample, 95th percentile): 49.79 ms
Inference Latency (single sample, 99th percentile): 64.55 ms
Inference Throughput (single sample): 32.35 FPS
Batch Throughput: 249.08 FPS

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 34.14 ms
Inference Latency (single sample, 95th percentile): 53.85 ms
Inference Latency (single sample, 99th percentile): 60.23 ms
Inference Throughput (single sample): 27.39 FPS
Batch Throughput: 281.65 FPS

-->

<!-- 

(Intel CPU)

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 12.69 ms
Inference Latency (single sample, 95th percentile): 12.83 ms
Inference Latency (single sample, 99th percentile): 12.97 ms
Inference Throughput (single sample): 78.73 FPS
Batch Throughput: 161.27 FPS

With compiling

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.47 ms
Inference Latency (single sample, 95th percentile): 8.58 ms
Inference Latency (single sample, 99th percentile): 8.79 ms
Inference Throughput (single sample): 117.86 FPS
Batch Throughput: 474.67 FPS



-->

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::


::: {.cell .markdown}

### Eager mode execution vs compiled model

We had just evaluated a model in eager mode. However, in some (although, not all) cases we may get better performance from compiling the model into a graph, and executing it as a graph.

Go back to the cell where the model is loaded, and add

```python
model.compile()
```

just below the call to `torch.load`. Then, run the notebook again ("Run > Run All Cells"). 

When you are done, download the fully executed notebook **again** from the Jupyter container environment for later reference.


:::
