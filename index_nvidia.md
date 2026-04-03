

# Model optimizations for serving

In this tutorial, we explore some model-level optimizations for model serving:

* graph optimizations
* quantization
* and hardware-specific execution providers, which switch out generic implementations of operations in the graph for hardware-specific optimized implementations

and we will see how these affect the throughput and inference time of a model.

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@UC and CHI@TACC sites.




## Context


The premise of this example is as follows: You are working as a machine learning engineer at a small startup. You have developed a two-stage image aesthetic scoring pipeline: a frozen CLIP ViT-L/14 vision encoder produces a 768-dimensional embedding, which is then fed through a lightweight MLP head (768 → 512 → 128 → 32 → 1 with sigmoid) that outputs a continuous aesthetic quality score from 0 to 1.

Now that you have trained the MLP head, you are preparing to serve predictions using this model. Your manager has advised that since you are an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763)
* inference on a server-grade GPU (NVIDIA A30 or A100)
* inference on end-user devices, as part of an app

You're already off to a good start, by using a lightweight MLP head on top of CLIP ViT-L/14 embeddings; the MLP is a small model that is especially well-suited for fast inference time. Now you need to measure the inference performance of the model and investigate ways to improve it.



## Experiment resources 

For this experiment, we will provision one bare-metal node with a recent NVIDIA GPU (e.g. A100, A30). (Although most of the experiment will run on CPU, we'll also do a little bit of GPU.)

We'll use the `compute_liqid` node types at CHI@TACC, or `compute_gigaio` node types at CHI@UC. (We won't use `compute_gigaio` nodes at CHI@TACC, which have a different GPU and CPU.)

* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs, and an AMD EPYC 7763 CPU.
* The `compute_gigaio` nodes at CHI@UC have an NVIDIA A100 80GB GPU, and an AMD EPYC 7763 CPU.

You can decide which type to use based on availability.



## Create a lease for a GPU server

To use bare metal resources on Chameleon, we must reserve them in advance. For this experiment, we will reserve a 3-hour block on a bare metal node with GPU.

We can use the OpenStack graphical user interface, Horizon, to submit a lease. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/)
* click "Experiment" > "CHI@TACC" or "Experiment" > "CHI@UC", depending on which site you want to make reservation at
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.



Then,

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `compute_liqid` or `compute_gigaio` as applicable to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone.
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, on the left side, click on the name of the node you want to reserve.
* Set the "Name" to `serve_model_netID`, replacing `netID` with your actual net ID.
* Set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
* Modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease ten minutes before the end of an hour, e.g. at `YY:50`.
* Click "Next".
* On the "Hosts" tab, confirm that the node you selected is listed in the "Resource properties" section, and click "Next".
* Then, click "Create". (We won't include any network resources in this lease.)

Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.



Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.



## At the beginning of your GPU server lease

At the beginning of your GPU lease time, you will continue with the next step, in which you bring up and configure a bare metal instance. To begin this step, open this experiment on Trovi:

* Use this link: [Model Optimizations for Aesthetic Score Prediction](https://chameleoncloud.org/experiment/share/c347ab71-1a5b-41cf-a2fd-0c34d30f1e1d) on Trovi
* Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it, including the notebook to bring up the bare metal server.

Inside the `model-serving-nvidia` directory, continue with `2_create_server.ipynb`.





## Launch and set up NVIDIA A100 or A30 server - with python-chi

At the beginning of the lease time for your bare metal server, we will bring up our GPU instance. We will use the `python-chi` Python API to Chameleon to provision our server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. Also **change the site to CHI@TACC or CHI@UC**, depending on where your reservation is.


```python
# runs in Chameleon Jupyter environment
from chi import server, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@TACC")
```


Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:


```python
# runs in Chameleon Jupyter environment
l = lease.get_lease(f"serve_model_netID") 
l.show()
```


The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting "Run" > "Run Selected Cell and All Below" from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!



We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image. 

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.



```python
# runs in Chameleon Jupyter environment
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-serve-model-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```


Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.



Then, we'll associate a floating IP with the instance, so that we can access it over SSH.


```python
# runs in Chameleon Jupyter environment
s.associate_floating_ip()
```

```python
# runs in Chameleon Jupyter environment
s.refresh()
s.check_connectivity()
```


In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).


```python
# runs in Chameleon Jupyter environment
s.refresh()
s.show(type="widget")
```





### Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.


```python
# runs in Chameleon Jupyter environment
s.execute("git clone -b main https://github.com/somadisingh/model-serving-nvidia")
```



### Set up Docker

To run the serving and inference experiments in this lab, we will use Docker containers that already include the required runtime libraries. In this step, we set up Docker on the server so we can launch those containers.


```python
# runs in Chameleon Jupyter environment
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```


### Set up the NVIDIA container toolkit


We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.


```python
# runs in Chameleon Jupyter environment
s.execute("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
# for https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
s.execute("sudo jq 'if has(\"exec-opts\") then . else . + {\"exec-opts\": [\"native.cgroupdriver=cgroupfs\"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json")
s.execute("sudo systemctl restart docker")
```



## Open an SSH session

Finally, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.

> **Docker group note**: If this is your first SSH session after running the setup cells above, close and reconnect once. The `docker` group membership added by `usermod` only takes effect in new login sessions.




## Prepare data

For this project, we use the Flickr-AES (ICCV 2017) aesthetic image scoring dataset, hosted on Google Drive. The dataset contains ~40K Flickr images with crowd-sourced aesthetic ratings, along with pre-computed train/val/test/production split manifests.

We'll prepare a Docker volume with this dataset so that the containers we create later can attach to it and access the data.




First, create the volume:

```bash
# runs on node-serve-model
docker volume create aesthetic_data
```

Then, to populate it with data, run

```bash
# runs on node-serve-model
docker compose -f model-serving-nvidia/docker/docker-compose-data.yaml up
```

This will run a temporary container that uses `gdown` to download the Flickr-AES images (~6 GB zip) from Google Drive, extracts them, and also downloads the split manifest CSVs. It may take several minutes depending on your connection speed. You can monitor progress in the terminal output, or verify completion with:

```bash
# runs on node-serve-model
docker ps
```

When there are no running containers, the download is complete.

Finally, verify that the data looks as it should:

```bash
# runs on node-serve-model
docker run --rm -it -v aesthetic_data:/mnt alpine sh -c "echo 'Images:' && ls /mnt/flickr-aes/40K/*.jpg | wc -l && echo 'Splits:' && ls /mnt/flickr-aes/splits/ && echo 'Inference:' && ls /mnt/flickr-aes/inference/images/ | wc -l && echo 'Inference personalized:' && ls /mnt/flickr-aes/inference_personalized/images/ | wc -l"
```

You should see ~40K images in the `40K/` folder, two CSV files in `splits/` (`flickr_global_manifest.csv` and `flickr_personalized_manifest.csv`), 7000 images in `inference/images/` (for global model benchmarking), and 5000 images in `inference_personalized/images/` (for personalized model benchmarking).



## Launch a Jupyter container

Inside the SSH session, build the `jupyter-onnx-gpu` image (includes CUDA, TensorRT, and all packages needed for notebooks 5–8):

```bash
# runs on node-serve-model
docker build -t jupyter-onnx-gpu -f model-serving-nvidia/docker/Dockerfile.jupyter-onnx-nvidia .
```

Then, launch a container from the `jupyter-onnx-gpu` image with GPU access:

```bash
# runs on node-serve-model
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/model-serving-nvidia/workspace:/home/jovyan/work/ \
    -v aesthetic_data:/mnt/ \
    -e AESTHETIC_DATA_DIR=/mnt/flickr-aes \
    --name jupyter \
    jupyter-onnx-gpu
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

Then, in the file browser on the left side, open the "work" directory and then click on the `5_measure_torch.ipynb` notebook to continue.



## Measure inference performance of PyTorch model on GPU 

First, we are going to measure the inference performance of an already-trained PyTorch model on CPU. Our full inference pipeline has two stages:

1. **CLIP ViT-L/14** (image encoder): Takes a raw image and produces a 768-dimensional embedding vector
2. **Aesthetic MLP head**: Takes the 768-dim embedding and produces an aesthetic quality score (0-1)

We will benchmark each stage independently, then measure the end-to-end pipeline. After completing this section, you should understand:

* how to measure the inference latency and throughput of a PyTorch model
* the relative cost of the image encoder (ViT) vs the downstream head (MLP)
* how to compare eager model execution vs a compiled model

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.


```python
# runs in jupyter container on node-serve-model
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import time
import numpy as np
import pandas as pd
import clip
```



## Resource monitoring

The `ResourceMonitor` class polls `nvidia-smi` (GPU utilization and memory) and `psutil` (CPU and RAM) in a background thread alongside each benchmark. The results tell you how much GPU, CPU, and RAM your workload actually needs — useful for right-sizing the instance.


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



First, let's load our MLP head and the CLIP ViT-L/14 model (used to compute image embeddings). We run all inference on the **GPU** — the A100's tensor cores make ViT encoding orders of magnitude faster than CPU.


```python
# runs in jupyter container on node-serve-model
model_path = "models/flickr_global_best_inference_only.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Load CLIP model for computing image embeddings
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
```


and also prepare our test dataset, using CLIP's own preprocessing:


```python
# runs in jupyter container on node-serve-model
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```



---

## Part 1: CLIP ViT-L/14 Image Encoder (GPU)

The ViT-L/14 model is the computationally expensive part of the pipeline. It processes raw images (224×224) through a Vision Transformer to produce 768-dimensional embeddings. Let's measure its performance on GPU in **eager mode** first.



#### ViT model size


```python
# runs in jupyter container on node-serve-model
# CLIP downloads the ViT model to ~/.cache/clip/
clip_cache_dir = os.path.expanduser("~/.cache/clip")
vit_model_file = os.path.join(clip_cache_dir, "ViT-L-14.pt")
if os.path.exists(vit_model_file):
    vit_model_size = os.path.getsize(vit_model_file)
    print(f"ViT-L/14 Model Size on Disk: {vit_model_size / (1e6):.2f} MB")
else:
    print(f"ViT model file not found at {vit_model_file}")
    # Estimate from parameters
    vit_params = sum(p.numel() * p.element_size() for p in clip_model.visual.parameters())
    print(f"ViT-L/14 Visual Encoder Size (in memory): {vit_params / (1e6):.2f} MB")
```


#### ViT single-image latency (eager mode)


```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Get a single preprocessed image
single_image, _ = next(iter(test_loader))
single_image = single_image[:1].to(device)

# Warm-up
with torch.no_grad():
    clip_model.encode_image(single_image)

monitor.start()
vit_latencies_eager = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        clip_model.encode_image(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_i = time.time() - start_time
        vit_latencies_eager.append(latencies_i)
monitor.stop()

print("ViT-L/14 Single Image Latency (Eager, GPU):")
print(f"  Median: {np.percentile(vit_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(vit_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(vit_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(vit_latencies_eager):.2f} FPS")
monitor.summary("ViT-L/14 eager single image (GPU)")
```


#### ViT batch throughput (eager mode)

We'll test with a batch of 32 images (matching our DataLoader batch size).


```python
# runs in jupyter container on node-serve-model
num_batches = 50
batch_images, _ = next(iter(test_loader))
batch_images = batch_images.to(device)

# Warm-up
with torch.no_grad():
    clip_model.encode_image(batch_images)

vit_batch_times_eager = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        clip_model.encode_image(batch_images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vit_batch_times_eager.append(time.time() - start_time)

vit_batch_fps_eager = (batch_images.shape[0] * num_batches) / np.sum(vit_batch_times_eager)
print(f"ViT-L/14 Batch Throughput (Eager, GPU, batch_size=32): {vit_batch_fps_eager:.2f} FPS")
```



#### ViT compiled mode (GPU)

Now let's compile the ViT visual encoder into a graph and see if we get a speedup. Graph compilation can fuse operations and optimize memory access patterns.


```python
# runs in jupyter container on node-serve-model
# Compile the visual encoder
clip_model.visual = torch.compile(clip_model.visual)

# Warm-up (first call triggers compilation, which is slow)
print("Compiling ViT visual encoder (this may take a moment)...")
with torch.no_grad():
    clip_model.encode_image(single_image)
print("Compilation complete.")
```

```python
# runs in jupyter container on node-serve-model
# Single-image latency (compiled)
vit_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        clip_model.encode_image(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vit_latencies_compiled.append(time.time() - start_time)

print("ViT-L/14 Single Image Latency (Compiled, GPU):")
print(f"  Median: {np.percentile(vit_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(vit_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(vit_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(vit_latencies_compiled):.2f} FPS")
```

```python
# runs in jupyter container on node-serve-model
# Batch throughput (compiled)
vit_batch_times_compiled = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        clip_model.encode_image(batch_images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vit_batch_times_compiled.append(time.time() - start_time)

vit_batch_fps_compiled = (batch_images.shape[0] * num_batches) / np.sum(vit_batch_times_compiled)
print(f"ViT-L/14 Batch Throughput (Compiled, GPU, batch_size=32): {vit_batch_fps_compiled:.2f} FPS")
```


#### ViT GPU summary


```python
# runs in jupyter container on node-serve-model
print("=" * 60)
print("CLIP ViT-L/14 GPU Benchmark Summary")
print("=" * 60)
print(f"{'Metric':<45} {'Eager':>8} {'Compiled':>8}")
print("-" * 60)
print(f"{'Single image latency (median, ms)':<45} {np.percentile(vit_latencies_eager, 50)*1000:>8.2f} {np.percentile(vit_latencies_compiled, 50)*1000:>8.2f}")
print(f"{'Single image latency (p95, ms)':<45} {np.percentile(vit_latencies_eager, 95)*1000:>8.2f} {np.percentile(vit_latencies_compiled, 95)*1000:>8.2f}")
print(f"{'Single image throughput (FPS)':<45} {num_trials/np.sum(vit_latencies_eager):>8.2f} {num_trials/np.sum(vit_latencies_compiled):>8.2f}")
print(f"{'Batch throughput (FPS, batch_size=32)':<45} {vit_batch_fps_eager:>8.2f} {vit_batch_fps_compiled:>8.2f}")
```



---

## Part 2: Aesthetic MLP Head (GPU)

Now we'll benchmark the lightweight MLP head that maps 768-dim CLIP embeddings to aesthetic scores. Since we'll compare eager vs compiled mode, we'll first reload the models fresh.



```python
# runs in jupyter container on node-serve-model
# Reload CLIP (uncompiled) and MLP for clean benchmarking
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()
```



#### MLP model size

Our `flickr_global_best_inference_only.pth` is a lightweight MLP head (768 → 512 → 128 → 32 → 1) that maps CLIP ViT-L/14 embeddings to aesthetic scores, so it is very small.

```python
# runs in jupyter container on node-serve-model
mlp_model_size = os.path.getsize(model_path) 
print(f"MLP Model Size on Disk: {mlp_model_size / (1e6):.2f} MB")
```


#### Sample predictions

Let's verify the model produces reasonable aesthetic scores.


```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    embeddings = torch.from_numpy(normalized(image_features.cpu().numpy())).float().to(device)
    scores = model(embeddings).squeeze()
    mean_score = scores.mean().item()
    std_score = scores.std().item()
```

```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (0-1):")
for i in range(min(5, len(scores))):
    print(f"  Image {i+1}: {scores[i].item():.2f}")
print(f"\nBatch mean: {mean_score:.2f}, std: {std_score:.2f}")
```


#### MLP inference latency (eager mode)

We pre-compute a CLIP embedding, then measure only the MLP forward pass.


```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Pre-compute a single CLIP embedding for benchmarking MLP latency
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    single_embedding = torch.from_numpy(normalized(sample_features.cpu().numpy())).float().to(device)

# Warm-up run 
with torch.no_grad():
    model(single_embedding)

mlp_latencies_eager = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(single_embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_latencies_eager.append(time.time() - start_time)
```

```python
# runs in jupyter container on node-serve-model
print("MLP Single Sample Latency (Eager, GPU):")
print(f"  Median: {np.percentile(mlp_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(mlp_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(mlp_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(mlp_latencies_eager):.2f} FPS")
```


#### MLP batch throughput (eager mode)


```python
# runs in jupyter container on node-serve-model
num_batches = 50

# Pre-compute a batch of CLIP embeddings for benchmarking MLP throughput
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    batch_embeddings = torch.from_numpy(normalized(batch_features.cpu().numpy())).float().to(device)

# Warm-up run 
with torch.no_grad():
    model(batch_embeddings)

mlp_batch_times_eager = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(batch_embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_batch_times_eager.append(time.time() - start_time)

mlp_batch_fps_eager = (batch_embeddings.shape[0] * num_batches) / np.sum(mlp_batch_times_eager)
print(f"MLP Batch Throughput (Eager, GPU, batch_size=32): {mlp_batch_fps_eager:.2f} FPS")
```



#### MLP compiled mode (GPU)


```python
# runs in jupyter container on node-serve-model
model = torch.compile(model)

# Warm-up (triggers compilation)
print("Compiling MLP model (this may take a moment)...")
with torch.no_grad():
    model(single_embedding)
print("Compilation complete.")
```

```python
# runs in jupyter container on node-serve-model
mlp_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(single_embedding)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_latencies_compiled.append(time.time() - start_time)

print("MLP Single Sample Latency (Compiled, GPU):")
print(f"  Median: {np.percentile(mlp_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(mlp_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(mlp_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(mlp_latencies_compiled):.2f} FPS")
```

```python
# runs in jupyter container on node-serve-model
mlp_batch_times_compiled = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(batch_embeddings)
        if device.type == "cuda":
            torch.cuda.synchronize()
        mlp_batch_times_compiled.append(time.time() - start_time)

mlp_batch_fps_compiled = (batch_embeddings.shape[0] * num_batches) / np.sum(mlp_batch_times_compiled)
print(f"MLP Batch Throughput (Compiled, GPU, batch_size=32): {mlp_batch_fps_compiled:.2f} FPS")
```



#### MLP GPU summary


```python
# runs in jupyter container on node-serve-model
print("=" * 60)
print("Aesthetic MLP Head GPU Benchmark Summary")
print("=" * 60)
print(f"MLP Model Size on Disk: {mlp_model_size / (1e6):.2f} MB")
print(f"Mean Predicted Score: {mean_score:.2f} (std: {std_score:.2f})")
print()
print(f"{'Metric':<45} {'Eager':>8} {'Compiled':>8}")
print("-" * 60)
print(f"{'Single sample latency (median, ms)':<45} {np.percentile(mlp_latencies_eager, 50)*1000:>8.2f} {np.percentile(mlp_latencies_compiled, 50)*1000:>8.2f}")
print(f"{'Single sample latency (p95, ms)':<45} {np.percentile(mlp_latencies_eager, 95)*1000:>8.2f} {np.percentile(mlp_latencies_compiled, 95)*1000:>8.2f}")
print(f"{'Single sample throughput (FPS)':<45} {num_trials/np.sum(mlp_latencies_eager):>8.2f} {num_trials/np.sum(mlp_latencies_compiled):>8.2f}")
print(f"{'Batch throughput (FPS, batch_size=32)':<45} {mlp_batch_fps_eager:>8.2f} {mlp_batch_fps_compiled:>8.2f}")
```


---

## Part 3: Personalized MLP Head (GPU)

The personalized model takes both a 768-dim CLIP embedding and a user index as input. It has an `nn.Embedding` table that maps each user index to a 64-dim learned vector, concatenates it with the CLIP embedding (832-dim total), and passes through the same MLP architecture. This lets the model learn per-user aesthetic preferences.



```python
# runs in jupyter container on node-serve-model
# Reload CLIP (uncompiled) for clean benchmarking
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Load personalized model
personal_model_path = "models/flickr_personalized_best_inference_only.pth"
personal_model = torch.load(personal_model_path, map_location=device, weights_only=False)
personal_model.eval()

# Get valid user indices from the personalized manifest
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
personal_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
seen_workers = sorted(personal_manifest.loc[personal_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique())
user2idx = {u: i for i, u in enumerate(seen_workers)}
num_users = len(user2idx)
print(f"Personalized model: {num_users} known users, embedding table shape: {personal_model.user_embedding.weight.shape}")
```


#### Personalized MLP model size


```python
# runs in jupyter container on node-serve-model
personal_mlp_model_size = os.path.getsize(personal_model_path)
print(f"Personalized MLP Model Size on Disk: {personal_mlp_model_size / (1e6):.2f} MB")
print(f"Global MLP Model Size on Disk:       {mlp_model_size / (1e6):.2f} MB")
```



#### Sample predictions

Let's verify the personalized model produces reasonable scores for a few different users on the same images.


```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    embeddings = torch.from_numpy(normalized(image_features.cpu().numpy())).float().to(device)

    # Pick 3 different users and score the same batch
    sample_user_ids = [0, num_users // 2, num_users - 1]
    for uid in sample_user_ids:
        user_idx = torch.full((embeddings.shape[0],), uid, dtype=torch.long, device=device)
        scores = personal_model(embeddings, user_idx).squeeze()
        print(f"User {uid}: mean={scores.mean().item():.3f}, std={scores.std().item():.3f}, first 3: {[f'{s:.3f}' for s in scores[:3].tolist()]}")
```



#### Personalized MLP inference latency (eager mode)

We pre-compute a CLIP embedding, then measure only the personalized MLP forward pass (embedding + user index → score).


```python
# runs in jupyter container on node-serve-model
num_trials = 100

# Pre-compute a single CLIP embedding for benchmarking
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    p_single_embedding = torch.from_numpy(normalized(sample_features.cpu().numpy())).float().to(device)
    p_single_user_idx = torch.tensor([0], dtype=torch.long, device=device)

# Warm-up run
with torch.no_grad():
    personal_model(p_single_embedding, p_single_user_idx)

personal_mlp_latencies_eager = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_single_embedding, p_single_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_latencies_eager.append(time.time() - start_time)
```

```python
# runs in jupyter container on node-serve-model
print("Personalized MLP Single Sample Latency (Eager, GPU):")
print(f"  Median: {np.percentile(personal_mlp_latencies_eager, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(personal_mlp_latencies_eager, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(personal_mlp_latencies_eager, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(personal_mlp_latencies_eager):.2f} FPS")
```



#### Personalized MLP batch throughput (eager mode)


```python
# runs in jupyter container on node-serve-model
num_batches = 50

# Pre-compute batch of CLIP embeddings
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    p_batch_embeddings = torch.from_numpy(normalized(batch_features.cpu().numpy())).float().to(device)
    p_batch_user_idx = torch.zeros(p_batch_embeddings.shape[0], dtype=torch.long, device=device)

# Warm-up run
with torch.no_grad():
    personal_model(p_batch_embeddings, p_batch_user_idx)

personal_mlp_batch_times_eager = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_batch_embeddings, p_batch_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_batch_times_eager.append(time.time() - start_time)

personal_mlp_batch_fps_eager = (p_batch_embeddings.shape[0] * num_batches) / np.sum(personal_mlp_batch_times_eager)
print(f"Personalized MLP Batch Throughput (Eager, GPU, batch_size=32): {personal_mlp_batch_fps_eager:.2f} FPS")
```



#### Personalized MLP compiled mode (GPU)


```python
# runs in jupyter container on node-serve-model
personal_model = torch.compile(personal_model)

# Warm-up (triggers compilation)
print("Compiling Personalized MLP model (this may take a moment)...")
with torch.no_grad():
    personal_model(p_single_embedding, p_single_user_idx)
print("Compilation complete.")
```

```python
# runs in jupyter container on node-serve-model
personal_mlp_latencies_compiled = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_single_embedding, p_single_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_latencies_compiled.append(time.time() - start_time)

print("Personalized MLP Single Sample Latency (Compiled, GPU):")
print(f"  Median: {np.percentile(personal_mlp_latencies_compiled, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(personal_mlp_latencies_compiled, 95) * 1000:.2f} ms")
print(f"  99th percentile: {np.percentile(personal_mlp_latencies_compiled, 99) * 1000:.2f} ms")
print(f"  Throughput: {num_trials/np.sum(personal_mlp_latencies_compiled):.2f} FPS")
```

```python
# runs in jupyter container on node-serve-model
personal_mlp_batch_times_compiled = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = personal_model(p_batch_embeddings, p_batch_user_idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        personal_mlp_batch_times_compiled.append(time.time() - start_time)

personal_mlp_batch_fps_compiled = (p_batch_embeddings.shape[0] * num_batches) / np.sum(personal_mlp_batch_times_compiled)
print(f"Personalized MLP Batch Throughput (Compiled, GPU, batch_size=32): {personal_mlp_batch_fps_compiled:.2f} FPS")
```



#### Personalized MLP GPU summary


```python
# runs in jupyter container on node-serve-model
print("=" * 65)
print("Personalized MLP Head GPU Benchmark Summary")
print("=" * 65)
print(f"Personalized MLP Model Size on Disk: {personal_mlp_model_size / (1e6):.2f} MB")
print()
print(f"{'Metric':<45} {'Eager':>8} {'Compiled':>8}")
print("-" * 65)
print(f"{'Single sample latency (median, ms)':<45} {np.percentile(personal_mlp_latencies_eager, 50)*1000:>8.2f} {np.percentile(personal_mlp_latencies_compiled, 50)*1000:>8.2f}")
print(f"{'Single sample latency (p95, ms)':<45} {np.percentile(personal_mlp_latencies_eager, 95)*1000:>8.2f} {np.percentile(personal_mlp_latencies_compiled, 95)*1000:>8.2f}")
print(f"{'Single sample throughput (FPS)':<45} {num_trials/np.sum(personal_mlp_latencies_eager):>8.2f} {num_trials/np.sum(personal_mlp_latencies_compiled):>8.2f}")
print(f"{'Batch throughput (FPS, batch_size=32)':<45} {personal_mlp_batch_fps_eager:>8.2f} {personal_mlp_batch_fps_compiled:>8.2f}")
```




---

## Part 4: End-to-End Pipeline (GPU)

Finally, let's measure the full pipeline: image → ViT → normalize → MLP → score. This shows the total latency a user would experience. We'll reload fresh (uncompiled) models.


```python
# runs in jupyter container on node-serve-model
# Reload uncompiled models for E2E measurement
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()
personal_model = torch.load(personal_model_path, map_location=device, weights_only=False)
personal_model.eval()
```

```python
# runs in jupyter container on node-serve-model
num_trials = 50

# Single image E2E
single_image = single_image[:1].to(device)

# Warm-up
with torch.no_grad():
    feat = clip_model.encode_image(single_image)
    emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
    model(emb)

monitor.start()
e2e_latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = model(emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_latencies.append(time.time() - start_time)
monitor.stop()

print("End-to-End Single Image Latency (GPU):")
print(f"  Median: {np.percentile(e2e_latencies, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_latencies, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_latencies):.2f} FPS")
monitor.summary("E2E pipeline single image (GPU)")
```

```python
# runs in jupyter container on node-serve-model
# Batch E2E
batch_images, _ = next(iter(test_loader))
batch_images = batch_images.to(device)

e2e_batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = model(emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_batch_times.append(time.time() - start_time)

e2e_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_batch_times)
print(f"End-to-End Batch Throughput (GPU, batch_size=32): {e2e_batch_fps:.2f} FPS")
```


#### End-to-End: Personalized MLP (GPU)

The personalized pipeline adds a user index lookup, but the ViT encode step is identical.


```python
# runs in jupyter container on node-serve-model
# Single image E2E - Personalized
p_user_idx_single = torch.tensor([0], dtype=torch.long, device=device)

# Warm-up
with torch.no_grad():
    feat = clip_model.encode_image(single_image)
    emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
    personal_model(emb, p_user_idx_single)

e2e_personal_latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(single_image)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = personal_model(emb, p_user_idx_single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_personal_latencies.append(time.time() - start_time)

print("End-to-End Single Image Latency - Personalized (GPU):")
print(f"  Median: {np.percentile(e2e_personal_latencies, 50) * 1000:.2f} ms")
print(f"  95th percentile: {np.percentile(e2e_personal_latencies, 95) * 1000:.2f} ms")
print(f"  Throughput: {num_trials / np.sum(e2e_personal_latencies):.2f} FPS")
```

```python
# runs in jupyter container on node-serve-model
# Batch E2E - Personalized
p_user_idx_batch = torch.zeros(batch_images.shape[0], dtype=torch.long, device=device)

e2e_personal_batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        feat = clip_model.encode_image(batch_images)
        emb = torch.from_numpy(normalized(feat.cpu().numpy())).float().to(device)
        _ = personal_model(emb, p_user_idx_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        e2e_personal_batch_times.append(time.time() - start_time)

e2e_personal_batch_fps = (batch_images.shape[0] * num_batches) / np.sum(e2e_personal_batch_times)
print(f"End-to-End Batch Throughput - Personalized (GPU, batch_size=32): {e2e_personal_batch_fps:.2f} FPS")
```

```python
# runs in jupyter container on node-serve-model
# Latency breakdown
vit_median = np.percentile(vit_latencies_eager, 50) * 1000
mlp_median = np.percentile(mlp_latencies_eager, 50) * 1000
personal_mlp_median = np.percentile(personal_mlp_latencies_eager, 50) * 1000
e2e_median = np.percentile(e2e_latencies, 50) * 1000
e2e_personal_median = np.percentile(e2e_personal_latencies, 50) * 1000

print("=" * 55)
print("End-to-End Latency Breakdown (GPU, single image)")
print("=" * 55)
print(f"  ViT encode:            {vit_median:.2f} ms ({vit_median/e2e_median*100:.1f}%)")
print(f"  Global MLP forward:    {mlp_median:.2f} ms ({mlp_median/e2e_median*100:.1f}%)")
print(f"  Personal MLP forward:  {personal_mlp_median:.2f} ms ({personal_mlp_median/e2e_personal_median*100:.1f}%)")
print(f"  E2E total (global):    {e2e_median:.2f} ms")
print(f"  E2E total (personal):  {e2e_personal_median:.2f} ms")
print()
print("The ViT encoder dominates the pipeline cost.")
print("Optimizing the MLP (ONNX, quantization, etc.) will")
print("improve MLP latency, but the total pipeline speedup")
print("depends primarily on the ViT encoder performance.")
```


---

## Part 5: Quality Metrics

Performance benchmarks tell you *how fast* the models run. Quality metrics tell you *how well* they predict — essential for right-sizing decisions (no point in ultra-low latency on a model that lacks accuracy).

We run both models over their respective held-out inference sets and compute:

**Global MLP**: MAE, RMSE, PLCC, SRCC, binary accuracy (threshold = 0.5), AUC-ROC  
**Personalized MLP**: same per-user metrics averaged across users, plus personalization gain vs global


```python
# runs in jupyter container on node-serve-model
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from PIL import Image
import glob

# ── helpers ──────────────────────────────────────────────────────────────────

def collect_global_predictions(manifest_df, image_root, clip_model, mlp_model, device,
                               clip_preprocess, batch_size=64):
    """Run the full ViT → MLP pipeline over all images in manifest_df.
    Returns (preds, targets) as numpy arrays."""
    preprocess = clip_preprocess
    preds, targets = [], []
    rows = manifest_df.reset_index(drop=True)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start + batch_size]
        imgs = []
        valid_mask = []
        for _, row in batch_rows.iterrows():
            img_path = os.path.join(image_root, row["image_name"])
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                imgs.append(img)
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
        if not imgs:
            continue
        img_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(img_tensor)
            embs = torch.from_numpy(normalized(feats.cpu().numpy())).float().to(device)
            scores = mlp_model(embs).squeeze().cpu().numpy()
        if scores.ndim == 0:
            scores = scores.reshape(1)
        gt = batch_rows.loc[[v for v, m in zip(batch_rows.index, valid_mask) if m], "global_score"].values
        preds.extend(scores.tolist())
        targets.extend(gt.tolist())

    return np.array(preds), np.array(targets)


def print_regression_metrics(preds, targets, label=""):
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    plcc, _ = pearsonr(preds, targets)
    srcc, _ = spearmanr(preds, targets)
    threshold = 0.5
    bin_acc = np.mean((preds >= threshold) == (targets >= threshold))
    try:
        auc = roc_auc_score((targets >= threshold).astype(int), preds)
    except ValueError:
        auc = float("nan")  # only one class present

    print(f"\n{'─'*55}")
    print(f"Quality metrics — {label}")
    print(f"{'─'*55}")
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  PLCC:             {plcc:.4f}")
    print(f"  SRCC:             {srcc:.4f}")
    print(f"  Binary accuracy:  {bin_acc:.4f}  (threshold={threshold})")
    print(f"  AUC-ROC:          {auc:.4f}")
    return dict(mae=mae, rmse=rmse, plcc=plcc, srcc=srcc, bin_acc=bin_acc, auc=auc)
```


### Global MLP — quality metrics

Load the test split from `flickr_global_manifest.csv` and run inference over all held-out images.


```python
# runs in jupyter container on node-serve-model
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
global_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_global_manifest.csv"))

# Use only the inference (test) split
global_test = global_manifest[global_manifest["split"] == "inference"].copy()
image_root_global = os.path.join(data_dir, "40K")

print(f"Global test set: {len(global_test)} images")
global_test.head()
```

```python
# runs in jupyter container on node-serve-model
# Reload clean models (without compile artifacts)
model_eval = torch.load("models/flickr_global_best_inference_only.pth",
                        map_location=device, weights_only=False)
model_eval.eval()
clip_eval, clip_pre_eval = clip.load("ViT-L/14", device=device)

print(f"Running GPU CLIP encoding over test set (~1-3 minutes)...")
global_preds, global_targets = collect_global_predictions(
    global_test, image_root_global, clip_eval, model_eval, device, clip_pre_eval
)
print(f"Collected {len(global_preds)} predictions")
global_metrics = print_regression_metrics(global_preds, global_targets, label="Global MLP")
```


### Personalized MLP — quality metrics

For the personalized model we compute metrics per user (using their held-out (image, score) pairs), then average across users.  
We also compute **personalization gain**: how much the personalized model improves on the global model for the same samples.


```python
# runs in jupyter container on node-serve-model
personal_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))

# Use only the inference split
personal_test = personal_manifest[personal_manifest["split"] == "inference"].copy()

# Use worker_score_norm as ground truth (already in [0,1])
personal_test = personal_test.rename(columns={"worker_score_norm": "global_score"})

# Build user index mapping (must match training)
seen_workers = sorted(
    personal_manifest.loc[personal_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique()
)
user2idx = {u: i for i, u in enumerate(seen_workers)}

image_root_personal = os.path.join(data_dir, "40K")
personal_model_eval = torch.load("models/flickr_personalized_best_inference_only.pth",
                                  map_location=device, weights_only=False)
personal_model_eval.eval()

print(f"Personalized test set: {len(personal_test)} rows, "
      f"{personal_test['worker_id'].nunique()} users")
```

```python
# runs in jupyter container on node-serve-model
per_user_srcc = []
per_user_mae  = []
all_personal_preds   = []
all_personal_targets = []
all_global_preds_on_personal = []

preprocess = clip_pre_eval
test_workers = [w for w in personal_test["worker_id"].unique() if w in user2idx]

for worker_id in test_workers:
    user_rows = personal_test[personal_test["worker_id"] == worker_id].reset_index(drop=True)
    if len(user_rows) < 3:
        continue  # skip users with too few samples for meaningful correlation
    uid = user2idx[worker_id]

    imgs, gt_scores = [], []
    for _, row in user_rows.iterrows():
        img_path = os.path.join(image_root_personal, row["image_name"])
        try:
            imgs.append(preprocess(Image.open(img_path).convert("RGB")))
            gt_scores.append(row["global_score"])
        except Exception:
            pass
    if not imgs:
        continue

    img_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = clip_eval.encode_image(img_tensor)
        embs  = torch.from_numpy(normalized(feats.cpu().numpy())).float().to(device)
        user_idx_tensor = torch.full((len(imgs),), uid, dtype=torch.long, device=device)
        p_scores = personal_model_eval(embs, user_idx_tensor).squeeze().cpu().numpy()
        g_scores = model_eval(embs).squeeze().cpu().numpy()

    if p_scores.ndim == 0:
        p_scores = p_scores.reshape(1)
    if g_scores.ndim == 0:
        g_scores = g_scores.reshape(1)

    gt = np.array(gt_scores)
    srcc_u, _ = spearmanr(p_scores, gt)
    mae_u     = np.mean(np.abs(p_scores - gt))
    per_user_srcc.append(srcc_u)
    per_user_mae.append(mae_u)
    all_personal_preds.extend(p_scores.tolist())
    all_personal_targets.extend(gt.tolist())
    all_global_preds_on_personal.extend(g_scores.tolist())

all_personal_preds          = np.array(all_personal_preds)
all_personal_targets        = np.array(all_personal_targets)
all_global_preds_on_personal = np.array(all_global_preds_on_personal)

print(f"Evaluated {len(test_workers)} users")
```

```python
# runs in jupyter container on node-serve-model
# Aggregate metrics
personal_metrics = print_regression_metrics(
    all_personal_preds, all_personal_targets, label="Personalized MLP (all users pooled)"
)

print(f"\n  Per-user SRCC (avg):  {np.mean(per_user_srcc):.4f}  "
      f"(std={np.std(per_user_srcc):.4f})")
print(f"  Per-user MAE  (avg):  {np.mean(per_user_mae):.4f}  "
      f"(std={np.std(per_user_mae):.4f})")

# Personalization gain
global_mae_on_personal = np.mean(np.abs(all_global_preds_on_personal - all_personal_targets))
personal_mae_on_personal = personal_metrics["mae"]
gain_mae = global_mae_on_personal - personal_mae_on_personal

global_srcc_on_personal, _ = spearmanr(all_global_preds_on_personal, all_personal_targets)
personal_srcc_on_personal, _ = spearmanr(all_personal_preds, all_personal_targets)
gain_srcc = personal_srcc_on_personal - global_srcc_on_personal

print(f"\n{'─'*55}")
print(f"Personalization gain (personalized vs global, same images)")
print(f"{'─'*55}")
print(f"  Global MAE  on personal set:  {global_mae_on_personal:.4f}")
print(f"  Personal MAE on personal set: {personal_mae_on_personal:.4f}")
print(f"  MAE improvement:              {gain_mae:+.4f}  {'✓ better' if gain_mae > 0 else '✗ worse'}")
print(f"  Global SRCC:                  {global_srcc_on_personal:.4f}")
print(f"  Personal SRCC:                {personal_srcc_on_personal:.4f}")
print(f"  SRCC improvement:             {gain_srcc:+.4f}  {'✓ better' if gain_srcc > 0 else '✗ worse'}")
```


When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)




## Measure inference performance of ONNX model on CPU 

To squeeze even more inference performance out of our model, we are going to convert it to ONNX format, which allows models from different frameworks (PyTorch, Tensorflow, Keras), to be deployed on a variety of different hardware platforms (CPU, GPU, edge devices), using many optimizations (graph optimizations, quantization, target device-specific implementations, and more).

After finishing this section, you should know:

* how to convert a PyTorch model to ONNX
* how to measure the inference latency and batch throughput of the ONNX model

and then you will use it to evaluate the optimized models you develop in the next section.

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.


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

```python
# runs in jupyter container on node-serve-model
# Prepare test dataset using CLIP's preprocessing
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```




First, let's load our saved PyTorch model, and convert it to ONNX using PyTorch's built-in `torch.onnx.export`:


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


## Create an inference session

Now, we can evaluate our model! To use an ONNX model, we create an *inference session*, and then use the model within that session.

For this first ONNX baseline, we will explicitly disable graph optimizations, so that later we can clearly see the effect when we enable them. Let's start an inference session:





```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
```

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




and let's double check the execution provider that will be used in this session:



```python
# runs in jupyter container on node-serve-model
ort_session.get_providers()
```





#### Sample predictions


First, let's verify the model produces reasonable aesthetic scores:


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

```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (0-1):")
for i in range(min(5, len(scores))):
    print(f"  Image {i+1}: {scores[i]:.2f}")
print(f"\nBatch mean: {mean_score:.2f}, std: {std_score:.2f}")
```


#### Model size

We are also concerned with the size of the ONNX model on disk. It will be similar to the equivalent PyTorch model size (to start!)


```python
# runs in jupyter container on node-serve-model
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```





#### Inference latency

Now, we'll measure how long it takes the model to return a prediction for a single sample. We will run 100 trials, and then compute aggregate statistics.


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



```python
# runs in jupyter container on node-serve-model
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```


#### Batch throughput 

Finally, we'll measure the rate at which the model can return predictions for batches of data. 


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

```python
# runs in jupyter container on node-serve-model
batch_fps = (batch_embeddings.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```




#### Summary of results


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


#### Quality metrics — Global FP32 ONNX baseline

These metrics show how well the **FP32 ONNX model** predicts aesthetic scores on the held-out test split. Establish this baseline before quantizing in notebook 7.


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



---

## Personalized MLP: ONNX Conversion and Baseline

The personalized model takes two inputs: a 768-dim CLIP embedding and a user index (integer). The user index is looked up in an `nn.Embedding` table to produce a 64-dim user vector, which is concatenated with the CLIP embedding before passing through the MLP. Let's convert it to ONNX and benchmark it.


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



#### Sample predictions


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

```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (personalized, 0-1):")
for i in range(min(5, len(p_scores))):
    print(f"  Image {i+1}: {p_scores[i]:.2f}")
print(f"\nBatch mean: {p_mean_score:.2f}, std: {p_std_score:.2f}")
```



#### Model size


```python
# runs in jupyter container on node-serve-model
personal_model_size = os.path.getsize(personal_onnx_path) 
print(f"Personalized Model Size on Disk: {personal_model_size / (1e6):.2f} MB")
print(f"Global Model Size on Disk:       {model_size / (1e6):.2f} MB")
```



#### Inference latency


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

```python
# runs in jupyter container on node-serve-model
print(f"Inference Latency (single sample, median): {np.percentile(p_latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(p_latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(p_latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(p_latencies):.2f} FPS")
```



#### Batch throughput


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

```python
# runs in jupyter container on node-serve-model
p_batch_fps = (p_batch_embeddings.shape[0] * num_batches) / np.sum(p_batch_times) 
print(f"Personalized Batch Throughput: {p_batch_fps:.2f} FPS")
```



#### Personalized ONNX summary


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


#### Quality metrics — Personalized FP32 ONNX baseline

Per-user SRCC and MAE across every annotator in the test split.


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



When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the `flickr_global.onnx` and `flickr_personalized.onnx` models from inside the `models` directory.




## Apply optimizations to ONNX model

Now that we have an ONNX model, we can apply some basic optimizations. After completing this section, you should be able to apply:

* graph optimizations, e.g. fusing operations
* post-training quantization (dynamic and static)
* and hardware-specific execution providers

to improve inference performance. 

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.




Since we are going to evaluate several models, we'll define a benchmark function here to help us compare them:



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
import pandas as pd
import clip
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from PIL import Image
```

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

```python
# runs in jupyter container on node-serve-model
# Prepare test dataset using CLIP's preprocessing
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

```python
# runs in jupyter container on node-serve-model
# Pre-compute CLIP embeddings for benchmarking
print("Pre-computing CLIP embeddings for benchmark data...")
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(clip_device))
    batch_embeddings = normalized(batch_features.cpu().numpy()).astype(np.float32)
    single_embedding = batch_embeddings[:1]
print(f"Embeddings shape: {batch_embeddings.shape}")
```

```python
# runs in jupyter container on node-serve-model
# Pre-compute FULL test embeddings for quality metrics (one-time cost; reused across all variants).
# The tiny MLP is then benchmarked 4 times on these embeddings — no CLIP re-run per variant.
print("Pre-computing test embeddings for quality metrics...")
print("(GPU CLIP encoding over the test set — should complete in 1-3 minutes.)")

_g_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_global_manifest.csv"))
_test_g = _g_manifest[_g_manifest["split"] == "inference"].reset_index(drop=True)
_img_root = os.path.join(data_dir, "40K")

_qm_g_embs_list, _qm_g_tgts = [], []
with torch.no_grad():
    for _i in range(0, len(_test_g), 32):
        _batch = _test_g.iloc[_i:_i+32]
        _imgs, _tgts = [], []
        for _, _row in _batch.iterrows():
            try:
                _imgs.append(clip_preprocess(Image.open(os.path.join(_img_root, _row["image_name"])).convert("RGB")))
                _tgts.append(_row["global_score"])
            except Exception:
                pass
        if not _imgs:
            continue
        _feats = clip_model.encode_image(torch.stack(_imgs).to(clip_device))
        _qm_g_embs_list.append(normalized(_feats.cpu().numpy()).astype(np.float32))
        _qm_g_tgts.extend(_tgts)
        if (_i // 32) % 20 == 0:
            print(f"  Global: {min(_i+32, len(_test_g))}/{len(_test_g)} ...")
_qm_g_embs = np.concatenate(_qm_g_embs_list, axis=0)
_qm_g_tgts = np.array(_qm_g_tgts, dtype=np.float32)

_p_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
_test_p = _p_manifest[_p_manifest["split"] == "inference"].reset_index(drop=True)
_seen_w = sorted(_p_manifest.loc[_p_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique())
_user2idx_qm = {u: i for i, u in enumerate(_seen_w)}

_qm_p_embs_list, _qm_p_tgts, _qm_p_uidxs = [], [], []
with torch.no_grad():
    for _i in range(0, len(_test_p), 32):
        _batch = _test_p.iloc[_i:_i+32]
        _imgs, _tgts, _uids = [], [], []
        for _, _row in _batch.iterrows():
            if _row["worker_id"] not in _user2idx_qm:
                continue
            try:
                _imgs.append(clip_preprocess(Image.open(os.path.join(_img_root, _row["image_name"])).convert("RGB")))
                _tgts.append(_row["worker_score_norm"])
                _uids.append(_user2idx_qm[_row["worker_id"]])
            except Exception:
                pass
        if not _imgs:
            continue
        _feats = clip_model.encode_image(torch.stack(_imgs).to(clip_device))
        _qm_p_embs_list.append(normalized(_feats.cpu().numpy()).astype(np.float32))
        _qm_p_tgts.extend(_tgts)
        _qm_p_uidxs.extend(_uids)
        if (_i // 32) % 20 == 0:
            print(f"  Personal: {min(_i+32, len(_test_p))}/{len(_test_p)} ...")
_qm_p_embs = np.concatenate(_qm_p_embs_list, axis=0)
_qm_p_tgts  = np.array(_qm_p_tgts,  dtype=np.float32)
_qm_p_uidxs = np.array(_qm_p_uidxs, dtype=np.int64)
print(f"Ready: {len(_qm_g_embs)} global, {len(_qm_p_embs)} personalized embeddings.")
```


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

    ## Quality metrics (on full test set using pre-computed embeddings)
    qm_preds = ort_session.run(None, {ort_session.get_inputs()[0].name: _qm_g_embs})[0].flatten()
    qm_mae  = np.mean(np.abs(qm_preds - _qm_g_tgts))
    qm_rmse = np.sqrt(np.mean((qm_preds - _qm_g_tgts) ** 2))
    qm_plcc, _ = pearsonr(qm_preds, _qm_g_tgts)
    qm_srcc, _ = spearmanr(qm_preds, _qm_g_tgts)
    qm_acc  = np.mean((qm_preds >= 0.5) == (_qm_g_tgts >= 0.5))
    qm_auc  = roc_auc_score((_qm_g_tgts >= 0.5).astype(int), qm_preds)
    print(f"\nQuality metrics (N={len(qm_preds)} — Global MLP):")
    print(f"  MAE: {qm_mae:.4f}  RMSE: {qm_rmse:.4f}  PLCC: {qm_plcc:.4f}  SRCC: {qm_srcc:.4f}  Acc: {qm_acc:.4f}  AUC: {qm_auc:.4f}")

```


```python
# runs in jupyter container on node-serve-model
# Prepare personalized model data
personal_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
seen_workers = sorted(personal_manifest.loc[personal_manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique())
num_users = len(seen_workers)

p_batch_user_idx = np.zeros(batch_embeddings.shape[0], dtype=np.int64)
p_single_user_idx = np.array([0], dtype=np.int64)

def benchmark_personal_session(ort_session):
    print(f"Execution provider: {ort_session.get_providers()}")

    ## Sample predictions
    input_names = [i.name for i in ort_session.get_inputs()]
    outputs = ort_session.run(None, {input_names[0]: batch_embeddings, input_names[1]: p_batch_user_idx})[0]
    scores = outputs.flatten()
    print(f"Sample scores (first 5): {', '.join(f'{s:.2f}' for s in scores[:5])}")
    print(f"Mean predicted score: {scores.mean():.2f}, Std: {scores.std():.2f}")

    ## Benchmark inference latency for single sample
    num_trials = 100
    ort_session.run(None, {input_names[0]: single_embedding, input_names[1]: p_single_user_idx})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {input_names[0]: single_embedding, input_names[1]: p_single_user_idx})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput
    num_batches = 50
    ort_session.run(None, {input_names[0]: batch_embeddings, input_names[1]: p_batch_user_idx})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {input_names[0]: batch_embeddings, input_names[1]: p_batch_user_idx})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_embeddings.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")

    ## Quality metrics (per-user SRCC and MAE on full test set)
    _qm_in = [i.name for i in ort_session.get_inputs()]
    qm_p_preds = ort_session.run(None, {_qm_in[0]: _qm_p_embs, _qm_in[1]: _qm_p_uidxs})[0].flatten()
    _per_srcc, _per_mae = [], []
    for uid in np.unique(_qm_p_uidxs):
        mask = _qm_p_uidxs == uid
        if mask.sum() < 3:
            continue
        _s, _ = spearmanr(qm_p_preds[mask], _qm_p_tgts[mask])
        _per_srcc.append(_s)
        _per_mae.append(np.mean(np.abs(qm_p_preds[mask] - _qm_p_tgts[mask])))
    print(f"\nQuality metrics (N={len(qm_p_preds)}, {len(_per_srcc)} users — Personalized MLP):")
    print(f"  Mean per-user SRCC: {np.mean(_per_srcc):.4f}  Mean per-user MAE: {np.mean(_per_mae):.4f}")

print(f"Personalized model: {num_users} known users")
```





### Apply basic graph optimizations

Let's start by applying some basic [graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode), e.g. fusing operations. 

We will save the model after applying graph optimizations to `models/flickr_global_optimized.onnx`, then evaluate that model in a new session.


```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
optimized_model_path = "models/flickr_global_optimized.onnx"

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED # apply graph optimizations
session_options.optimized_model_filepath = optimized_model_path 

ort_session = ort.InferenceSession(onnx_model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
```



Download the `flickr_global_optimized.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `flickr_global.onnx` and review the graph. Then, upload the `flickr_global_optimized.onnx` and see what has changed in the "optimized" graph.




Next, evaluate the optimized model. The graph optimizations may improve the inference performance, may have negligible effect, OR they can make it worse, depending on the model and the hardware environment in which the model is executed.



```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_optimized.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```

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


#### Personalized MLP: Graph optimizations

Apply the same graph optimizations to the personalized model.


```python
# runs in jupyter container on node-serve-model
personal_onnx_path = "models/flickr_personalized.onnx"
personal_optimized_path = "models/flickr_personalized_optimized.onnx"

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session_options.optimized_model_filepath = personal_optimized_path

ort_session = ort.InferenceSession(personal_onnx_path, sess_options=session_options, providers=['CPUExecutionProvider'])
print(f"Personalized optimized model saved to {personal_optimized_path}")
```

```python
# runs in jupyter container on node-serve-model
personal_optimized_path = "models/flickr_personalized_optimized.onnx"
ort_session = ort.InferenceSession(personal_optimized_path, providers=['CPUExecutionProvider'])
benchmark_personal_session(ort_session)
```



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





Post-training quantization comes in two main types. In both types, FP32 values will be converted in INT8, using

$$\texttt{val}\_\texttt{quant} = \texttt{round}\left(\frac{\texttt{val}\_\texttt{fp32}}{\texttt{scale}}\right) + \texttt{zero}\_\texttt{point}$$

but they differ with respect to when and how the quantization parameters "scale" and "zero point" are computed:

* dynamic quantization: weights are quantized in advance and stored in INT8 representation. The quantization parameters for the activations are computed during inference. 
* static quantization: weights are quantized in advance and stored in INT8, and the quantization parameters are also set in advance for activations. This approach requires the use of a "calibration dataset" during quantization, to set the quantization parameters for the activations.

 




#### Dynamic quantization

We will start with dynamic quantization. No calibration dataset is required. 
 

```python
# runs in jupyter container on node-serve-model
import neural_compressor
from neural_compressor import quantization
```

```python
# runs in jupyter container on node-serve-model
# Load ONNX model into Intel Neural Compressor
model_path = "models/flickr_global.onnx"
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
q_model.save_model_to_file("models/flickr_global_quantized_dynamic.onnx")
```



Download the `flickr_global_quantized_dynamic.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `flickr_global.onnx` and review the graph. Then, upload the `flickr_global_quantized_dynamic.onnx` and see what has changed in the quantized graph.

Note that some of our operations have become integer operations, but we have added additional operations to quantize and dequantize activations throughout the graph. 



We are also concerned with the size of the quantized model on disk:


```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_quantized_dynamic.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```




Next, evaluate the quantized model. Since we are saving weights in integer form, the model size is smaller. With respect to inference time, however, while the integer operations may be faster than their FP32 equivalents, the dynamic quantization and dequantization of activations may add more compute time than we save from integer operations.



```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_quantized_dynamic.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```


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



#### Personalized MLP: Dynamic quantization

Apply dynamic quantization to the personalized model.


```python
# runs in jupyter container on node-serve-model
personal_fp32 = neural_compressor.model.onnx_model.ONNXModel("models/flickr_personalized.onnx")
config_ptq = neural_compressor.PostTrainingQuantConfig(approach="dynamic")
p_q_model = quantization.fit(model=personal_fp32, conf=config_ptq)
p_q_model.save_model_to_file("models/flickr_personalized_quantized_dynamic.onnx")
```

```python
# runs in jupyter container on node-serve-model
p_dyn_size = os.path.getsize("models/flickr_personalized_quantized_dynamic.onnx")
print(f"Personalized Quantized (Dynamic) Size on Disk: {p_dyn_size / (1e6):.2f} MB")
```

```python
# runs in jupyter container on node-serve-model
ort_session = ort.InferenceSession("models/flickr_personalized_quantized_dynamic.onnx", providers=['CPUExecutionProvider'])
benchmark_personal_session(ort_session)
```



#### Static quantization


Next, we will try static quantization with a calibration dataset. 

First, let's prepare the calibration dataset by pre-computing CLIP embeddings from the validation images. Since the ONNX model expects 768-dim embedding inputs, our calibration data must be in the same format.


```python
# runs in jupyter container on node-serve-model
import neural_compressor
from neural_compressor import quantization
```

```python
# runs in jupyter container on node-serve-model
# Pre-compute CLIP embeddings for calibration/evaluation
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print("Pre-computing CLIP embeddings for calibration data...")
cal_embeddings = []
with torch.no_grad():
    for images, _ in val_loader:
        features = clip_model.encode_image(images.to(clip_device))
        embs = normalized(features.cpu().numpy()).astype(np.float32)
        cal_embeddings.append(embs)
cal_embeddings = np.vstack(cal_embeddings)
print(f"Calibration embeddings shape: {cal_embeddings.shape}")

# Wrap embeddings in a dataset for INC
cal_dataset = TensorDataset(torch.from_numpy(cal_embeddings), torch.zeros(len(cal_embeddings)))
cal_dataloader = neural_compressor.data.DataLoader(framework='onnxruntime', dataset=cal_dataset)
```


Then, we'll configure the quantizer. We'll start with a more aggressive quantization strategy, quantizing as much as possible.




```python
# runs in jupyter container on node-serve-model
# Load ONNX model into Intel Neural Compressor
model_path = "models/flickr_global.onnx"
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
q_model.save_model_to_file("models/flickr_global_quantized_aggressive.onnx")
```


Download the `flickr_global_quantized_aggressive.onnx` model from inside the `models` directory. 


To see the effect of the graph optimizations, we can visualize the models using [Netron](https://netron.app/). Upload the original `flickr_global.onnx` and review the graph. Then, upload the `flickr_global_quantized_aggressive.onnx` and see what has changed in the quantized graph.

Note that within the parameters for each quantized operation, we now have a "scale" and "zero point" - these are used to convert the FP32 values to INT8 values, as described above. The optimal scale and zero point for weights is determined by the fitted weights themselves, but the calibration dataset was required to find the optimal scale and zero point for activations.





Let's get the size of the quantized model on disk:


```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_quantized_aggressive.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```




Next, evaluate the quantized model.


```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_quantized_aggressive.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```



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


Let's try a more conservative approach to static quantization next, with a lower quantization level. With `quant_level=0`, fewer operations are quantized, which typically preserves more of the original model's output fidelity at the cost of less compression.



```python
# runs in jupyter container on node-serve-model
# Load ONNX model into Intel Neural Compressor
model_path = "models/flickr_global.onnx"
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
q_model.save_model_to_file("models/flickr_global_quantized_conservative.onnx")
```


Download the `flickr_global_quantized_conservative.onnx` model from inside the `models` directory. 


To see the effect of the quantization, we can visualize the models using [Netron](https://netron.app/). Upload the `flickr_global_quantized_conservative.onnx` and see what has changed in the quantized graph, relative to the "aggressive quantization" graph.

In this graph, since only some operations are quantized, we have a "Quantize" node before each quantized operation in the graph, and a "Dequantize" node after.






Let's get the size of the quantized model on disk:


```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_quantized_conservative.onnx"
model_size = os.path.getsize(onnx_model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```




Next, evaluate the quantized model. While we see some savings in model size relative to the unquantized model, the additional quantize and dequantize operations can make the inference time much slower.

However, these tradeoffs vary from one model to the next, and across implementations and hardware. In some cases, the quantize-dequantize model may still have faster inference times than the unquantized models.




```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global_quantized_conservative.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```



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



### Quantization aware training

To achieve the best of both worlds - high accuracy, but the small model size and faster inference time of a quantized model - we can try quantization aware training. In QAT, the effect of quantization is "simulated" during training, so that we learn weights that are more robust to quantization. Then, when we quantize the model, we can achieve better accuracy.


-->




---

### Personalized MLP: Static Quantization

Apply the same static quantization approaches to the personalized model. We need a calibration dataloader with two inputs (embedding + user_idx).


```python
# runs in jupyter container on node-serve-model
# Prepare calibration dataloader for personalized model (two inputs)
p_cal_user_idx = np.zeros(len(cal_embeddings), dtype=np.int64)
p_cal_dataset = TensorDataset(
    torch.from_numpy(cal_embeddings), 
    torch.from_numpy(p_cal_user_idx)
)
p_cal_dataloader = neural_compressor.data.DataLoader(framework='onnxruntime', dataset=p_cal_dataset)
```


#### Personalized MLP: Aggressive static quantization


```python
# runs in jupyter container on node-serve-model
personal_fp32 = neural_compressor.model.onnx_model.ONNXModel("models/flickr_personalized.onnx")

config_ptq = neural_compressor.PostTrainingQuantConfig(
    approach="static", device='cpu', quant_level=1,
    quant_format="QOperator",
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"},
    calibration_sampling_size=128
)

p_q_model = quantization.fit(
    model=personal_fp32, conf=config_ptq, calib_dataloader=p_cal_dataloader
)
p_q_model.save_model_to_file("models/flickr_personalized_quantized_aggressive.onnx")
```

```python
# runs in jupyter container on node-serve-model
p_agg_size = os.path.getsize("models/flickr_personalized_quantized_aggressive.onnx")
print(f"Personalized Quantized (Aggressive) Size on Disk: {p_agg_size / (1e6):.2f} MB")
```

```python
# runs in jupyter container on node-serve-model
ort_session = ort.InferenceSession("models/flickr_personalized_quantized_aggressive.onnx", providers=['CPUExecutionProvider'])
benchmark_personal_session(ort_session)
```



#### Personalized MLP: Conservative static quantization


```python
# runs in jupyter container on node-serve-model
personal_fp32 = neural_compressor.model.onnx_model.ONNXModel("models/flickr_personalized.onnx")

config_ptq = neural_compressor.PostTrainingQuantConfig(
    approach="static", device='cpu', quant_level=0,
    quant_format="QOperator",
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"},
    calibration_sampling_size=128
)

p_q_model = quantization.fit(
    model=personal_fp32, conf=config_ptq, calib_dataloader=p_cal_dataloader
)
p_q_model.save_model_to_file("models/flickr_personalized_quantized_conservative.onnx")
```

```python
# runs in jupyter container on node-serve-model
p_cons_size = os.path.getsize("models/flickr_personalized_quantized_conservative.onnx")
print(f"Personalized Quantized (Conservative) Size on Disk: {p_cons_size / (1e6):.2f} MB")
```

```python
# runs in jupyter container on node-serve-model
ort_session = ort.InferenceSession("models/flickr_personalized_quantized_conservative.onnx", providers=['CPUExecutionProvider'])
benchmark_personal_session(ort_session)
```



When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

Also download the models from inside the `models` directory.




### GPU Inference: ViT Encoder and ONNX Execution Providers

In this notebook, we will:

1. Benchmark the **CLIP ViT-L/14 image encoder on GPU** (eager + compiled, across multiple batch sizes)
2. Measure the **end-to-end pipeline on GPU** (image → ViT → MLP → score)
3. Test the **MLP head with different ONNX execution providers** (CPU, CUDA, TensorRT, OpenVINO)

You are already running in the `jupyter-onnx-gpu` container that was launched in notebook 4 — no container switch is needed for Parts 1, 2, and 3 (CUDA and TensorRT execution providers).

> **Note**: The OpenVINO execution provider requires a separate container. Instructions for switching are provided in the OpenVINO section later in this notebook.




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


## Resource monitoring

The `ResourceMonitor` class polls `nvidia-smi` (GPU utilization and memory) and `psutil` (CPU and RAM) in a background thread. It runs alongside each execution provider benchmark so you can see exactly how much GPU memory and compute each EP consumes — the primary signal for right-sizing.


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
data_dir = os.getenv("AESTHETIC_DATA_DIR", "flickr-aes")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'inference'), transform=clip_preprocess)
```




---

## Part 1: CLIP ViT-L/14 on GPU

The ViT is the heavy part of the pipeline. On GPU, we can process much larger batches efficiently. We'll benchmark across multiple batch sizes to see how throughput scales and find the point where the GPU is fully utilized.



#### ViT GPU: Eager mode across batch sizes


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



#### ViT GPU: Compiled mode across batch sizes

Now let's compile the ViT visual encoder for potential further speedup on GPU.


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


#### ViT GPU summary


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



---

## Part 2: End-to-End Pipeline on GPU

Let's measure the full pipeline on GPU: image → ViT (GPU) → normalize → MLP (CPU) → score. The MLP stays on CPU since it's tiny and the overhead of GPU kernel launch would likely dominate.


```python
# runs in jupyter container on node-serve-model
# Reload uncompiled CLIP for clean E2E measurement
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# Load MLP on CPU
mlp_model = torch.load("models/flickr_global_best_inference_only.pth", map_location=torch.device("cpu"), weights_only=False)
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



### Personalized MLP: End-to-End Pipeline

We repeat the end-to-end measurement with the **PersonalizedMLP**, which takes an additional **user index** input alongside the CLIP embedding.


```python
# runs in jupyter container on node-serve-model
# Load personalized model and prepare user indices
personal_model = torch.load("models/flickr_personalized_best_inference_only.pth", map_location="cpu", weights_only=False)
personal_model.eval()

manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_personalized_manifest.csv"))
seen_workers = sorted(manifest[manifest["worker_split"] == "seen_worker_pool"]["worker_id"].unique())
user2idx = {u: i for i, u in enumerate(seen_workers)}
print(f"Personalized model loaded. Number of seen users: {len(seen_workers)}")

# Use first user for benchmarking
sample_user_idx = torch.tensor([0], dtype=torch.long)
```

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



---

## Part 3: MLP ONNX Execution Providers

Now we'll benchmark the MLP head using different ONNX Runtime execution providers. Since the MLP takes pre-computed 768-dim embeddings as input, we pre-compute those once and then time only the ONNX inference.


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






#### CPU execution provider

First, for reference, we'll run the MLP ONNX model with the `CPUExecutionProvider`:





```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)
```

```python
# runs in jupyter container on node-serve-model
# Personalized MLP - CPU execution provider
personal_onnx_path = "models/flickr_personalized.onnx"
ort_session = ort.InferenceSession(personal_onnx_path, providers=['CPUExecutionProvider'])
benchmark_personal_session(ort_session)
```

<!-- placeholder: update with real benchmark numbers -->



#### CUDA execution provider

Next, we'll try the CUDA execution provider, which will execute the MLP model on the GPU:





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

<!-- placeholder: update with real benchmark numbers -->



#### Pre-compute test embeddings for TRT precision check

TensorRT on Ampere GPUs may silently apply FP16 precision, which can degrade prediction quality. We pre-compute all test embeddings once on GPU (fast: seconds), then verify quality metrics after each TRT benchmark call.


```python
# runs in jupyter container on node-serve-model
# Pre-compute all test embeddings on GPU (one-time; used for TRT quality checks below)
print("Pre-computing test embeddings on GPU for TRT quality verification...")
_trt_g_manifest = pd.read_csv(os.path.join(data_dir, "splits", "flickr_global_manifest.csv"))
_trt_test_g = _trt_g_manifest[_trt_g_manifest["split"] == "inference"].reset_index(drop=True)
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
_trt_test_p = _trt_p_manifest[_trt_p_manifest["split"] == "inference"].reset_index(drop=True)
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



#### TensorRT execution provider


The TensorRT execution provider will optimize the model for inference on NVIDIA GPUs. It will take a long time to run this cell, because it spends a lot of time optimizing the model (finding the best subgraphs, etc.) - but once the model is loaded, its inference time will be much faster than any of our previous tests.




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
personal_onnx_path = "models/flickr_personalized.onnx"
monitor.start()
ort_session = ort.InferenceSession(personal_onnx_path, providers=['TensorrtExecutionProvider'])
benchmark_personal_session(ort_session)
monitor.stop()
monitor.summary("Personalized MLP — TensorrtExecutionProvider")
ort.get_device()
```

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


```python
# runs in jupyter container on node-serve-model
onnx_model_path = "models/flickr_global.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['OpenVINOExecutionProvider'])
benchmark_session(ort_session)
ort.get_device()
```

```python
# runs in jupyter container on node-serve-model
# Personalized MLP - OpenVINO execution provider
personal_onnx_path = "models/flickr_personalized.onnx"
ort_session = ort.InferenceSession(personal_onnx_path, providers=['OpenVINOExecutionProvider'])
benchmark_personal_session(ort_session)
ort.get_device()
```


<!-- placeholder: update with real benchmark numbers -->


When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)


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
docker compose -f ~/model-serving-nvidia/docker/docker-compose-fastapi.yaml up -d
```

4. Get the Jupyter token:

```bash
# runs on node-serve-model
docker exec jupyter_fastapi jupyter server list
```

5. Open this notebook in the Jupyter container at `http://<FLOATING_IP>:8888`.


```python
import requests
import time
import numpy as np
import concurrent.futures
```


## Resource monitoring

The `ResourceMonitor` class polls `psutil` (CPU and RAM) in a background thread during benchmarks. Because the Jupyter container in this setup does not have direct GPU access, **server-side GPU metrics** are best observed on the host with:

```bash
# runs on node-serve-model (in a separate terminal)
watch -n 1 nvidia-smi
# or
docker stats fastapi_server
```


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


## Health check

Verify the FastAPI server is reachable.


```python
FASTAPI_URL = "http://fastapi_server:8000"

resp = requests.get(f"{FASTAPI_URL}/health")
print(resp.status_code, resp.json())
```


## Prepare test embeddings

We create random 768-dim embeddings (L2-normalized) to simulate CLIP outputs. The MLP model doesn't care about semantic content for latency benchmarking.


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


---

## Part 1: Global MLP — Single Request Latency

Send sequential requests one at a time and measure round-trip latency.


```python
url = f"{FASTAPI_URL}/predict/global"
payload = {"embedding": single_emb.tolist()}

# Quick sanity check
resp = requests.post(url, json=payload)
print(f"Status: {resp.status_code}, Response: {resp.json()}")
```

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


## Part 2: Global MLP — Batch Request Latency

Send a batch of 32 embeddings in a single request.


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


## Part 3: Global MLP — Concurrent Requests

Simulate multiple clients sending concurrent single requests. This tests queuing behavior.


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


---

## Part 4: Personalized MLP

Repeat the same measurements for the personalized model endpoint, which takes an additional `user_idx` input.


```python
personal_url = f"{FASTAPI_URL}/predict/personalized"
personal_payload = {"embedding": single_emb.tolist(), "user_idx": 0}

# Sanity check
resp = requests.post(personal_url, json=personal_payload)
print(f"Status: {resp.status_code}, Response: {resp.json()}")
```

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


---

## Summary

After running all cells above, fill in the results:


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


When you are done, download the fully executed notebook for later reference.

Then, bring down the FastAPI service:

```bash
# runs on node-serve-model
docker compose -f ~/model-serving-nvidia/docker/docker-compose-fastapi.yaml down
```


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


```python
import numpy as np
import time
import tritonclient.http as httpclient
```


## Resource monitoring

The `ResourceMonitor` class polls `psutil` (CPU and RAM) in a background thread. Because the Jupyter container in this setup does not have direct GPU access, **server-side GPU metrics** can be pulled from Triton's built-in Prometheus metrics endpoint on port 8002, or observed on the host:

```bash
# runs on node-serve-model (in a separate terminal)
watch -n 1 nvidia-smi
# or
docker stats triton_server
```


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


## Server-side GPU metrics via Triton

Triton exposes live GPU and inference statistics in Prometheus format at `http://triton_server:8002/metrics`. The cell below pulls the key GPU metrics — utilization, memory, and per-model inference count — directly from that endpoint.


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


---

## Part 1: Verify Triton is ready and models are loaded


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


## Prepare test data


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


---

## Part 2: Global MLP — Triton Client Benchmark

### Sanity check


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


### Sequential single-sample latency


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


### Batch throughput (batch_size=32)


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


---

## Part 3: Personalized MLP — Triton Client Benchmark

### Sanity check


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


```python
# Placeholder: paste perf_analyzer results here for reference
# Concurrency=1:  Avg latency = ____ usec (queue=____, compute infer=____), throughput=____ infer/sec
# Concurrency=8:  Avg latency = ____ usec (queue=____, compute infer=____), throughput=____ infer/sec
# Concurrency=16: Avg latency = ____ usec (queue=____, compute infer=____), throughput=____ infer/sec
```


### Batch-size sweep

Test different batch sizes with a single concurrent client:

```bash
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 1  --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 8  --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 16 --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 32 --shape input:768 --concurrency-range 1
docker exec triton_server perf_analyzer -u localhost:8000 -m flickr_global -b 64 --shape input:768 --concurrency-range 1
```


```python
# Placeholder: paste batch-size sweep results
# b=1:  throughput=____ infer/sec, latency=____ usec
# b=8:  throughput=____ infer/sec, latency=____ usec
# b=16: throughput=____ infer/sec, latency=____ usec
# b=32: throughput=____ infer/sec, latency=____ usec
# b=64: throughput=____ infer/sec, latency=____ usec
```


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


```python
# Placeholder: paste scaling results
# 1 instance, concurrency=8:  throughput=____, queue=____ usec
# 2 instances, concurrency=8: throughput=____, queue=____ usec
```


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


```python
# Placeholder: paste dynamic batching results
# Rate=200 (no batching):  avg latency=____ usec, queue=____ usec
# Rate=200 (batching):     avg latency=____ usec, queue=____ usec
# Rate=500 (batching):     avg latency=____ usec, queue=____ usec
```


---

## Summary


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


When you are done, download the fully executed notebook for later reference.

Then, bring down the Triton service:

```bash
# runs on node-serve-model
docker compose -f ~/model-serving-nvidia/docker/docker-compose-triton.yaml down
```



<hr>

<small>Questions about this material? Contact  Somaditya</small>

<hr>

<small>This material is based upon work supported by the National Somadi Foundation under Grant No. 17tsapt2f.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Somadi Foundation.</small>