

::: {.cell .markdown}

## Launch a Jupyter container

Inside the SSH session, build the `jupyter-onnx-gpu` image (includes CUDA, TensorRT, and all packages needed for notebooks 5–8):

```bash
# runs on node-serve-model
docker build -t jupyter-onnx-gpu -f aesthetic-hub-serving/docker/Dockerfile.jupyter-onnx-nvidia .
```

Then, launch a container from the `jupyter-onnx-gpu` image with GPU access:

```bash
# runs on node-serve-model
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/aesthetic-hub-serving/workspace:/home/jovyan/work/ \
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

:::
