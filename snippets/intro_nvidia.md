
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


The premise of this example is as follows: You are working as a machine learning engineer at a small startup. You have developed a two-stage image aesthetic scoring pipeline: a frozen CLIP ViT-L/14 vision encoder produces a 768-dimensional embedding, which is then fed through a lightweight MLP head (768 → 512 → 128 → 32 → 1 with sigmoid) that outputs a continuous aesthetic quality score from 0 to 1.

Now that you have trained the MLP head, you are preparing to serve predictions using this model. Your manager has advised that since you are an early-stage startup, they can't afford much compute for serving models. Your manager wants you to prepare a few different options, that they will then price out among cloud providers and decide which to use:

* inference on a server-grade CPU (AMD EPYC 7763)
* inference on a server-grade GPU (NVIDIA A30 or A100)
* inference on end-user devices, as part of an app

You're already off to a good start, by using a lightweight MLP head on top of CLIP ViT-L/14 embeddings; the MLP is a small model that is especially well-suited for fast inference time. Now you need to measure the inference performance of the model and investigate ways to improve it.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one bare-metal node with a recent NVIDIA GPU (e.g. A100, A30). (Although most of the experiment will run on CPU, we'll also do a little bit of GPU.)

We'll use the `compute_liqid` node types at CHI@TACC, or `compute_gigaio` node types at CHI@UC. (We won't use `compute_gigaio` nodes at CHI@TACC, which have a different GPU and CPU.)

* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs, and an AMD EPYC 7763 CPU.
* The `compute_gigaio` nodes at CHI@UC have an NVIDIA A100 80GB GPU, and an AMD EPYC 7763 CPU.

You can decide which type to use based on availability.

:::
