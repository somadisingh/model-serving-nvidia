
::: {.cell .markdown}

# Model optimizations for serving

In this tutorial, we explore some model-level optimizations for model serving:

* graph optimizations
* quantization
* and hardware-specific execution providers, which switch out generic implementations of operations in the graph for hardware-specific optimized implementations

and we will see how these affect the throughput and inference time of a model.

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@UC and CHI@TACC sites.

:::


::: {.cell .markdown}

## Context


The premise of this example is as follows: You are working as a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.

Now that you have trained a model, you are preparing to serve predictions using this model. Your manager has advised that since GourmetGram is an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763). Your manager wants to see an option that has less than 3ms median inference latency for a single input sample, and has a batch throughput of at least 1000 frames per second.
* inference on a server-grade GPU (NVIDIA A30 or A100). Since GourmetGram won't be able to afford to load balance across several GPUs, your manager said that the GPU option must have strong enough performance to handle the workload with a single GPU node: they are looking for less than 1ms median inference latency for a single input sample, and a batch throughput of at least 5000 frames per second.
* inference on end-user devices, as part of an app. For this option, the model itself should be less than 5MB on disk, because users are sensitive to storage space on mobile devices. Because the total prediction time will not include any network delay when the model is on the end-user device, the "budget" for inference time is larger: your manager wants less than 15ms median inference latency for a single input sample on a low-resource edge device (ARM Cortex A76 processor).

You're already off to a good start, by using a lightweight MLP head on top of CLIP ViT-L/14 embeddings; the MLP is a small model that is especially well-suited for fast inference time. Now you need to measure the inference performance of the model and, if it doesn't meet the requirements above, investigate ways to improve it.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one bare-metal node with a recent NVIDIA GPU (e.g. A100, A30). (Although most of the experiment will run on CPU, we'll also do a little bit of GPU.)

We'll use the `compute_liqid` node types at CHI@TACC, or `compute_gigaio` node types at CHI@UC. (We won't use `compute_gigaio` nodes at CHI@TACC, which have a different GPU and CPU.)

* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs, and an AMD EPYC 7763 CPU.
* The `compute_gigaio` nodes at CHI@UC have an NVIDIA A100 80GB GPU, and an AMD EPYC 7763 CPU.

You can decide which type to use based on availability.

:::
