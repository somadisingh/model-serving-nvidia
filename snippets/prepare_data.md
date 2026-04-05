

::: {.cell .markdown}

## Prepare data

For this project, we use the Flickr-AES (ICCV 2017) aesthetic image scoring dataset, hosted on Google Drive. The dataset contains ~40K Flickr images with crowd-sourced aesthetic ratings, along with pre-computed train/val/test/production split manifests.

We'll prepare a Docker volume with this dataset so that the containers we create later can attach to it and access the data.

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
docker compose -f aesthetic-hub-serving/docker/docker-compose-data.yaml up
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

:::