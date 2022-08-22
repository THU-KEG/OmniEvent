# Benchmark

## Comparison between Hugging Face Transformers

### Make Big Models trainable on consumer GPUs

Tested on 32GB V100 machine using `bert-large-uncased`, we have comparable throughput as Hugging Face Transformers but much fewer GPU memory footprint.

|repo|**max**-batchsize(#examples)|time(s)|throughput(#examples/s)|
|-|-|-|-|
|transformers|11|1.11|9.9|
|transformers+fp16|14|0.53|26.4|
|modelcenter|256|10.3|24.9|

Tested on a **single consumer GPU**, 11GB 2080Ti, however, training `bert-large-uncased` is no longer supported in Hugging Face Transformers, but we make it possible.

|repo|**max**-batchsize(#examples)|
|-|-|
|transformers|0|
|transformers+fp16|0|
|modelcenter|72|

### Make Huge Models train easily.

Tested on 40GB A100 machine using `T5-11B`, we make it possible to train with 16 batch-size using two GPUs.

## Comparison between Deepspeed ZeRO

see also [BMTrain's Performance](https://github.com/OpenBMB/BMTrain#performance)