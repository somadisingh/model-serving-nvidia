

::: {.cell .markdown}

## Prepare data

For the rest of this tutorial, we'll use an aesthetic image scoring dataset hosted on [HuggingFace](https://huggingface.co/datasets/somadisingh/aesthetic-hub). We're going to prepare a Docker volume with this dataset already prepared on it, so that the containers we create later can attach to this volume and access the data. 

:::


::: {.cell .markdown}

First, create the volume:

```bash
# runs on node-serve-model
docker volume create aesthetic_data
```

Then, to populate it with data, run

```bash
# runs on node-serve-model
docker compose -f serve-model-chi/docker/docker-compose-data.yaml up -d
```

This will run a temporary container that downloads the aesthetic scoring dataset from HuggingFace, extracts it in the volume, and then stops. It may take a few minutes depending on your connection speed (the dataset is ~3.3 GB). You can verify with 

```bash
# runs on node-serve-model
docker ps
```

that it is done - when there are no running containers.

Finally, verify that the data looks as it should. Start a shell in a temporary container with this volume attached, and `ls` the contents of the volume:

```bash
# runs on node-serve-model
docker run --rm -it -v aesthetic_data:/mnt alpine ls -l /mnt/aesthetic-hub/
```

it should show "test" and "validation" subfolders, along with "metadata.csv" and "tags.csv".

:::