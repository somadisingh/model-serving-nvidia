# Model optimizations for serving machine learning models

In this tutorial, we explore some model-level optimizations for model serving:

* graph optimizations
* quantization
* and hardware-specific execution providers, which switch out generic implementations of operations in the graph for hardware-specific optimized implementations

and we will see how these affect the throughput and inference time of a model. Follow along at [Model optimizations for serving machine learning models](https://teaching-on-testbeds.github.io/serve-model-chi).

Note: this tutorial requires advance reservation of specific hardware! You can use

* The `compute_gigaio` nodes at CHI@UC , which have an NVIDIA A100 80GB GPU, and an AMD EPYC 7763 CPU.
