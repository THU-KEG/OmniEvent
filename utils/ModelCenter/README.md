<div align="center">

<h1><img src="docs/source/_static/images/logo.png" height="32px"/> ModelCenter</h1>

**Efficient Low-Resource Implementations of Big Models**

</div>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-models">Supported Models</a> •
  <a href="./README-ZH.md" target="_blank">简体中文</a>
</p>

<p align="center">

<a href='https://modelcenter.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/modelcenter/badge/?version=latest' alt='Documentation Status' />
</a>

<a href="https://github.com/OpenBMB/ModelCenter/releases">
    <img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/OpenBMB/ModelCenter?include_prereleases">
</a>

<a href="https://github.com/OpenBMB/ModelCenter/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/ModelCenter">
</a>

</p>

## What's New
- 2022/07/14 [**ModelCenter 0.1.4**]() ModelCenter supports Mengzi, GLM, Longformer, and KV_PLM.
- 2022/07/05 [**ModelCenter 0.1.3**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.1.3) ModelCenter supports mT5, T5v1.1, ViT, and Wenzhong.
- 2022/04/27 [**ModelCenter 0.1.1**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.1.1) ModelCenter supports RoBERTa. 
- 2022/04/06 [**ModelCenter 0.1.0**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.1.0) ModelCenter has publicly released the first stable version, which fixes some bugs in model performance and GPU memory usage.
- 2022/03/21 [**ModelCenter 0.0.1-beta**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.0.1-beta) ModelCenter has publicly released the first beta version.

## Overview

ModelCenter implements pre-trained language models (PLMs) based on the backend [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain/). ModelCenter supports Efficient, Low-Resource, Extendable model usage and distributed training.

Our main advantages are:

- **Easy to use**. Compared to Deepspeed and Megatron, we have better and more flexible code-packaging and easy to configure python environments, and the training code is uniform with PyTorch style.
- **More efficient memory utilization**. Models with large memory footprints can cause OOM (out of memory) before the computational power of the GPU is fully utilized. Our implementation reduces the memory footprint by several times, allowing more efficient use of the GPU's computational power with a larger batch size.
- **Efficient distributed training with low resources**. With the support of [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain/), we are able to easily extend the [ZeRO](https://ieeexplore.ieee.org/abstract/document/9355301) optimization to any PLMs, and we optimize communication and time scheduling for faster distributed training.

## Documentation

Our [documentation](https://modelcenter.readthedocs.io/) provides more information about the package.

## Installation

### 1. From PyPI (Recommend)

```shell
$ pip install model-center
```

### 2. From Source

```shell
$ git clone https://github.com/OpenBMB/ModelCenter.git
$ cd ModelCenter
$ pip install -r requirements.txt
$ python3 setup.py install
```

## Quick Start

In the quick start, you will walk through how to fine-tune a [BERT](https://arxiv.org/abs/1810.04805) model on a classification task.

### 1. Initialize bmtrain backend
First, you need to import `bmtrain` and use `bmtrain.init_distributed()` at the beginning of your code, which can initialize the distributed environments. 

```python
import bmtrain as bmt
bmt.init_distributed(seed=0)
```

### 2. Prepare the model
Next, you can simply get a pre-trained BERT model from `model_center`, e.g., *bert-base-uncased*. When fine-tuning BERT on the classification task, a feed-forward layer need to be appended to the last layer.

```python
import torch
from model_center.model import Bert, BertConfig
from model_center.layer import Linear

class BertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = Bert.from_pretrained("bert-base-uncased")
        self.dense = Linear(config.dim_model, 2)
        bmt.init_parameters(self.dense)

    def forward(self, input_ids, attention_mask):
        pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)
```

If only config is needed instead of pretrained checkpoint, you can initialize a model as the following:

```python
config = BertConfig.from_json_file("your/path/to/config.json")
model = Bert(config)
bmt.init_parameters(model)
# bmt.load(model, "your/path/to/pytorch_model.pt")
```

### 3. Perpare the dataset
The next step is to prepare the dataset used for training and evaluation. Here, we use the [BoolQ](https://github.com/google-research-datasets/boolean-questions) dataset from the [SuperGLUE benchmark](https://super.gluebenchmark.com/). You need to download the dataset and put the unzipped folder in `your_path_to_dataset`.

```python
from model_center.dataset.bertdataset import DATASET
from model_center.dataset import DistributedDataLoader
from model_center.tokenizer import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
splits = ['train', 'dev']
dataset = {}

for split in splits:
    dataset[split] = DATASET['BoolQ']('your_path_to_dataset', split, bmt.rank(), bmt.world_size(), tokenizer, max_encoder_length=512)

batch_size = 64
train_dataloader = DistributedDataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
dev_dataloader = DistributedDataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
```

### 4. Train the model
Now, select optimizer, learning rate scheduler, loss function, and then start training the model! Here, we train BERT for 5 epochs and evaluate it at the end of each epoch.

```python
optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())

lr_scheduler = bmt.lr_scheduler.Noam(
    optimizer, 
    start_lr = 1e-5,
    warmup_iter = 100, 
    end_iter = -1)

loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

for epoch in range(5):
    model.train()
    for data in train_dataloader:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']

        optimizer.zero_grad()

        # model forward
        logits = model(input_ids, attention_mask)

        # calculate loss
        loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # use bmt.sum_loss(loss) to gather all loss information from all distributed processes
        global_loss = bmt.sum_loss(loss).item()

        # scale loss to avoid precision underflow of fp16
        loss = optimizer.loss_scale(loss)

        # model backward
        loss.backward()

        # clip gradient norm
        grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale = optimizer.scale, norm_type = 2)

        bmt.optim_step(optimizer, lr_scheduler)

        # print information only on rank 0 when distributed training
        bmt.print_rank(
            "loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                global_loss,
                lr_scheduler.current_lr,
                int(optimizer.scale),
                grad_norm,
            )
        )

    # evaluate model
    model.eval()
    with torch.no_grad():
        pd = [] # prediction
        gt = [] # ground_truth
        for data in dev_dataloader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

            logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(labels.cpu().tolist())

        # gather results from all distributed processes
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        # calculate metric
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(gt, pd)
        bmt.print_rank(f"accuracy: {acc*100:.2f}")
```

### 5. Run your code

You can run the above code using the same launch command as the distributed module of PyTorch.

Choose one of the following commands depending on your version of PyTorch.

* `${MASTER_ADDR}` means the IP address of the master node.
* `${MASTER_PORT}` means the port of the master node.
* `${NNODES}` means the total number of nodes.
* `${GPU_PER_NODE}` means the number of GPUs per node.
* `${NODE_RANK}` means the rank of this node.

#### torch.distributed.launch (more suitable for torch < 1.10)

```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} \
                                      --master_port ${MASTER_PORT} \
                                      --nproc_per_node ${GPU_PER_NODE} \
                                      --nnodes ${NNODES} \
                                      --node_rank ${NODE_RANK} \
                                      train.py
```

#### torchrun (more suitable for torch >= 1.10)

```shell
$ torchrun --nnodes=${NNODES} \
           --nproc_per_node=${GPU_PER_NODE} \
           --rdzv_id=1 \
           --rdzv_backend=c10d \
           --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
           train.py
```

More information can be found from the [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility).


## Supported Models


- CPM-1[[paper](https://www.sciencedirect.com/science/article/pii/S266665102100019X)]. We currently support loading the following checkpoint via ``CPM1.from_pretrained(identifier)`` as follows,

    - cpm1-large

- CPM-2[[paper](https://www.sciencedirect.com/science/article/pii/S2666651021000310)]. We currently support loading the following checkpoint via ``CPM2.from_pretrained(identifier)`` as follows,

    - cpm2-large

- BERT[[paper](https://arxiv.org/abs/1810.04805)]. We currently support loading the following checkpoint via ``Bert.from_pretrained(identifier)`` as follows,

    - bert-base-cased
    - bert-base-uncased
    - bert-large-cased
    - bert-large-uncased
    - bert-base-chinese
    - bert-base-multilingual-cased
    - kv-plm

- RoBERTa[[paper](https://arxiv.org/abs/1907.11692)]. We currently support loading the following checkpoint via ``Roberta.from_pretrained(identifier)`` of the following:

    - roberta-base
    - roberta-large

- T5[[paper](https://arxiv.org/abs/1910.10683)]. We currently support loading the following checkpoint via ``T5.from_pretrained(identifier)`` of the following:

    - t5-small
    - t5-base
    - t5-large
    - t5-3b
    - t5-11b
    - t5-v1_1-small
    - t5-v1_1-base
    - t5-v1_1-large
    - t5-v1_1-xl
    - t5-v1_1-xxl
    - mt5-small
    - mt5-base
    - mt5-large
    - mt5-xl
    - mt5-xxl
    - mengzi-t5-base

- GPT-2[[paper](http://www.persagen.com/files/misc/radford2019language.pdf)]. We currently support loading the following checkpoint via ``GPT2.from_pretrained(identifier)`` of the following:

    - gpt2-base
    - gpt2-medium
    - gpt2-large
    - gpt2-xl
    - wenzhong-gpt2-3.5b

- GPT-J[[paper](https://github.com/kingoflolz/mesh-transformer-jax)]. We currently support loading the following checkpoint via ``GPTj.from_pretrained(identifier)`` of the following:

    - gptj-6b
 
- Longformer[[paper](https://arxiv.org/abs/2004.05150)]. we currently support loading the following checkpoint via `` Longformer.from_pretrained(identifier)``  of the following:

    - lawformer

- GLM[[paper](https://arxiv.org/abs/2103.10360)]. we currently support loading the following checkpoint via ``GLM.from_pretrained(identifier)`` of the following:

    - glm-10b-zh

- ViT[[paper](https://arxiv.org/abs/2010.11929)]. we currently support loading the following checkpoint via `` ViT.from_pretrained(identifier)``  of the following:

    - vit-base-patch16-224 

## Performance

You can find more performance metrics in the repo [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain).

## Community

We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/ModelCenter/blob/main/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 735930538
- Website: https://www.openbmb.org
- Weibo: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## License

The package is released under the [Apache 2.0](https://github.com/OpenBMB/ModelCenter/blob/main/LICENSE) License.

