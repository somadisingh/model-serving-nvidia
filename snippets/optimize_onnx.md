

::: {.cell .markdown}

## Apply optimizations to ONNX model

Now that we have an ONNX model, we can apply some basic optimizations. After completing this section, you should be able to apply:

* graph optimizations, e.g. fusing operations
* post-training quantization (dynamic and static)
* and hardware-specific execution providers

to improve inference performance. 

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.

:::


::: {.cell .markdown}

Since we are going to evaluate several models, we'll define a benchmark function here to help us compare them:

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
from torch.utils.data import DataLoader, TensorDataset
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
data_dir = os.getenv("AESTHETIC_DATA_DIR", "aesthetic-hub")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Pre-compute CLIP embeddings for benchmarking
print("Pre-computing CLIP embeddings for benchmark data...")
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    batch_embeddings = normalized(batch_features.cpu().numpy()).astype(np.float32)
    single_embedding = batch_embeddings[:1]
print(f"Embeddings shape: {batch_embeddings.shape}")
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

### Apply basic graph optimizations

Let's start by applying some basic [graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode), e.g. fusing operations. 

We will save the model after applying graph optimizations to `models/aesthetic_mlp_optimized.onnx`, then evaluate that model in a new session.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp.onnx"
optimized_model_path = "models/aesthetic_mlp_optimized.onnx"

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED # apply graph optimizations
session_options.optimized_model_filepath = optimized_model_path 

ort_session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
```
:::


::: {.cell .markdown}

Download the `aesthetic_mlp_optimized.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `aesthetic_mlp.onnx` and review the graph. Then, upload the `aesthetic_mlp_optimized.onnx` and see what has changed in the "optimized" graph.

:::


::: {.cell .markdown}

Next, evaluate the optimized model. The graph optimizations may improve the inference performance, may have negligible effect, OR they can make it worse, depending on the model and the hardware environment in which the model is executed.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_optimized.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```
:::

<!--

On gigaio AMD EPYC:


Execution provider: ['CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.70 ms
Inference Latency (single sample, 95th percentile): 8.88 ms
Inference Latency (single sample, 99th percentile): 9.24 ms
Inference Throughput (single sample): 114.63 FPS
Batch Throughput: 1153.63 FPS

On liqid Intel:

Execution provider: ['CPUExecutionProvider']
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 4.63 ms
Inference Latency (single sample, 95th percentile): 4.67 ms
Inference Latency (single sample, 99th percentile): 4.75 ms
Inference Throughput (single sample): 214.45 FPS
Batch Throughput: 2488.54 FPS

-->

::: {.cell .markdown}

### Apply post training quantization

We will continue our quest to improve inference speed! The next optimization we will attempt is quantization.

There are many frameworks that offer quantization - for our aesthetic MLP model, we could:

* use [PyTorch quantization](https://docs.pytorch.org/ao/stable/index.html)
* use [ONNX quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
* use [Intel Neural Compressor](https://intel.github.io/neural-compressor/latest/index.html) (which supports PyTorch and ONNX models)
* use [NNCF](https://github.com/openvinotoolkit/nncf) if we plan to use the OpenVINO execution provider
* etc...

These frameworks vary in the type of quantization they support, the range of operations that may be quantized, and many other details.

We will use Intel Neural Compressor, which in addition to supporting many ML frameworks and many types of quantization has an interesting feature: it supports quantization up to a specified evaluation threshold. In other words, we can specify "quantize as much as possible, but without losing more than 0.01 accuracy" and Intel Neural Compressor will find the best quantized version of the model that does not lose more than 0.01 accuracy.

:::


::: {.cell .markdown}


Post-training quantization comes in two main types. In both types, FP32 values will be converted in INT8, using

$$\texttt{val}\_\texttt{quant} = \texttt{round}\left(\frac{\texttt{val}\_\texttt{fp32}}{\texttt{scale}}\right) + \texttt{zero}\_\texttt{point}$$

but they differ with respect to when and how the quantization parameters "scale" and "zero point" are computed:

* dynamic quantization: weights are quantized in advance and stored in INT8 representation. The quantization parameters for the activations are computed during inference. 
* static quantization: weights are quantized in advance and stored in INT8, and the quantization parameters are also set in advance for activations. This approach requires the use of a "calibration dataset" during quantization, to set the quantization parameters for the activations.

 
:::



::: {.cell .markdown}

#### Dynamic quantization

We will start with dynamic quantization. No calibration dataset is required. 
 
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import neural_compressor
from neural_compressor import quantization
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load ONNX model into Intel Neural Compressor
model_path = "models/aesthetic_mlp.onnx"
fp32_model = neural_compressor.model.onnx_model.ONNXModel(model_path)

# Configure the quantizer
config_ptq = neural_compressor.PostTrainingQuantConfig(
    approach="dynamic"
)

# Fit the quantized model
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq
)

# Save quantized model
q_model.save_model_to_file("models/aesthetic_mlp_quantized_dynamic.onnx")
```
:::


::: {.cell .markdown}

Download the `aesthetic_mlp_quantized_dynamic.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `aesthetic_mlp.onnx` and review the graph. Then, upload the `aesthetic_mlp_quantized_dynamic.onnx` and see what has changed in the quantized graph.

Note that some of our operations have become integer operations, but we have added additional operations to quantize and dequantize activations throughout the graph. 

:::

::: {.cell .markdown}

We are also concerned with the size of the quantized model on disk:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_quantized_dynamic.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::



::: {.cell .markdown}

Next, evaluate the quantized model. Since we are saving weights in integer form, the model size is smaller. With respect to inference time, however, while the integer operations may be faster than their FP32 equivalents, the dynamic quantization and dequantization of activations may add more compute time than we save from integer operations.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_quantized_dynamic.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```
:::


<!-- 

On liqid AMD EPYC

Model Size on Disk: 2.42 MB
Execution provider: ['CPUExecutionProvider']
Accuracy: 82.04% (2746/3347 correct)
Inference Latency (single sample, median): 22.32 ms
Inference Latency (single sample, 95th percentile): 22.97 ms
Inference Latency (single sample, 99th percentile): 23.14 ms
Inference Throughput (single sample): 44.71 FPS
Batch Throughput: 38.34 FPS

On liqid Intel

Execution provider: ['CPUExecutionProvider']
Accuracy: 84.58% (2831/3347 correct)
Inference Latency (single sample, median): 28.29 ms
Inference Latency (single sample, 95th percentile): 29.00 ms
Inference Latency (single sample, 99th percentile): 29.07 ms
Inference Throughput (single sample): 35.28 FPS

-->


::: {.cell .markdown}

#### Static quantization


Next, we will try static quantization with a calibration dataset. 

First, let's prepare the calibration dataset by pre-computing CLIP embeddings from the validation images. Since the ONNX model expects 768-dim embedding inputs, our calibration data must be in the same format.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import neural_compressor
from neural_compressor import quantization
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Pre-compute CLIP embeddings for calibration/evaluation
data_dir = os.getenv("AESTHETIC_DATA_DIR", "aesthetic-hub")
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'validation'), transform=clip_preprocess)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print("Pre-computing CLIP embeddings for calibration data...")
cal_embeddings = []
with torch.no_grad():
    for images, _ in val_loader:
        features = clip_model.encode_image(images.to(device))
        embs = normalized(features.cpu().numpy()).astype(np.float32)
        cal_embeddings.append(embs)
cal_embeddings = np.vstack(cal_embeddings)
print(f"Calibration embeddings shape: {cal_embeddings.shape}")

# Wrap embeddings in a dataset for INC
cal_dataset = TensorDataset(torch.from_numpy(cal_embeddings), torch.zeros(len(cal_embeddings)))
cal_dataloader = neural_compressor.data.DataLoader(framework='onnxruntime', dataset=cal_dataset)
```
:::

::: {.cell .markdown}

Then, we'll configure the quantizer. We'll start with a more aggressive quantization strategy, quantizing as much as possible.


:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load ONNX model into Intel Neural Compressor
model_path = "models/aesthetic_mlp.onnx"
fp32_model = neural_compressor.model.onnx_model.ONNXModel(model_path)

# Configure the quantizer - aggressive static quantization
config_ptq = neural_compressor.PostTrainingQuantConfig(
    approach="static", 
    device='cpu', 
    quant_level=1,
    quant_format="QOperator", 
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"}, 
    calibration_sampling_size=128
)

# Fit the quantized model using calibration data
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq, 
    calib_dataloader=cal_dataloader
)

# Save quantized model
q_model.save_model_to_file("models/aesthetic_mlp_quantized_aggressive.onnx")
```
:::

::: {.cell .markdown}

Download the `aesthetic_mlp_quantized_aggressive.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `aesthetic_mlp.onnx` and review the graph. Then, upload the `aesthetic_mlp_quantized_aggressive.onnx` and see what has changed in the quantized graph.

Note that within the parameters for each quantized operation, we now have a "scale" and "zero point" - these are used to convert the FP32 values to INT8 values, as described above. The optimal scale and zero point for weights is determined by the fitted weights themselves, but the calibration dataset was required to find the optimal scale and zero point for activations.

:::



::: {.cell .markdown}

Let's get the size of the quantized model on disk:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_quantized_aggressive.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::



::: {.cell .markdown}

Next, evaluate the quantized model.
:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_quantized_aggressive.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```
:::



<!-- 

On AMD EPYC

Model Size on Disk: 2.42 MB
Accuracy: 87.12% (2916/3347 correct)
Inference Latency (single sample, median): 7.52 ms
Inference Latency (single sample, 95th percentile): 7.78 ms
Inference Latency (single sample, 99th percentile): 7.84 ms
Inference Throughput (single sample): 132.40 FPS
Batch Throughput: 899.98 FPS

Model Size on Disk: 2.42 MB
Accuracy: 87.12% (2916/3347 correct)
Inference Latency (single sample, median): 7.85 ms
Inference Latency (single sample, 95th percentile): 8.14 ms
Inference Latency (single sample, 99th percentile): 8.26 ms
Inference Throughput (single sample): 126.58 FPS
Batch Throughput: 739.48 FPS

On Intel

Execution provider: ['CPUExecutionProvider']
Accuracy: 89.87% (3008/3347 correct)
Inference Latency (single sample, median): 2.51 ms
Inference Latency (single sample, 95th percentile): 2.60 ms
Inference Latency (single sample, 99th percentile): 2.71 ms
Inference Throughput (single sample): 396.18 FPS
Batch Throughput: 2057.18 FPS


-->

::: {.cell .markdown}

Let's try a more conservative approach to static quantization next, with a lower quantization level. With `quant_level=0`, fewer operations are quantized, which typically preserves more of the original model's output fidelity at the cost of less compression.

:::


::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
# Load ONNX model into Intel Neural Compressor
model_path = "models/aesthetic_mlp.onnx"
fp32_model = neural_compressor.model.onnx_model.ONNXModel(model_path)

# Configure the quantizer - conservative static quantization
config_ptq = neural_compressor.PostTrainingQuantConfig(
    approach="static", 
    device='cpu', 
    quant_level=0,  # 0 is a less aggressive quantization level
    quant_format="QOperator", 
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"}, 
    calibration_sampling_size=128
)

# Fit the quantized model
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq, 
    calib_dataloader=cal_dataloader
)

# Save quantized model
q_model.save_model_to_file("models/aesthetic_mlp_quantized_conservative.onnx")
```
:::

::: {.cell .markdown}

Download the `aesthetic_mlp_quantized_conservative.onnx` model from inside the `models` directory. 


To see the effect of the quantization, we can visualize the models using [Netron](https://netron.app/). Upload the `aesthetic_mlp_quantized_conservative.onnx` and see what has changed in the quantized graph, relative to the "aggressive quantization" graph.

In this graph, since only some operations are quantized, we have a "Quantize" node before each quantized operation in the graph, and a "Dequantize" node after.

:::




::: {.cell .markdown}

Let's get the size of the quantized model on disk:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_quantized_conservative.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::



::: {.cell .markdown}

Next, evaluate the quantized model. While we see some savings in model size relative to the unquantized model, the additional quantize and dequantize operations can make the inference time much slower.

However, these tradeoffs vary from one model to the next, and across implementations and hardware. In some cases, the quantize-dequantize model may still have faster inference times than the unquantized models.
:::




::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/aesthetic_mlp_quantized_conservative.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```
:::



<!--

on AMD EPYC

Model Size on Disk: 6.01 MB
Accuracy: 90.20% (3019/3347 correct)
Inference Latency (single sample, median): 10.20 ms
Inference Latency (single sample, 95th percentile): 10.39 ms
Inference Latency (single sample, 99th percentile): 10.66 ms
Inference Throughput (single sample): 97.87 FPS
Batch Throughput: 277.23 FPS

On intel

Execution provider: ['CPUExecutionProvider']
Accuracy: 90.44% (3027/3347 correct)
Inference Latency (single sample, median): 6.60 ms
Inference Latency (single sample, 95th percentile): 6.66 ms
Inference Latency (single sample, 99th percentile): 6.68 ms
Inference Throughput (single sample): 151.36 FPS
Batch Throughput: 540.19 FPS

-->

<!--


::: {.cell .markdown}

### Quantization aware training

To achieve the best of both worlds - high accuracy, but the small model size and faster inference time of a quantized model - we can try quantization aware training. In QAT, the effect of quantization is "simulated" during training, so that we learn weights that are more robust to quantization. Then, when we quantize the model, we can achieve better accuracy.

:::

-->



::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the models from inside the `models` directory.

:::
