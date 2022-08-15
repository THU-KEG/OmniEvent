# Quick start

In the quick start, you will walk through how to fine-tune a [BERT](https://arxiv.org/abs/1810.04805) model on a classification task.

## Initialize bmtrain backend
First, you need to import `bmtrain` and use `bmtrain.init_distributed()` at the beginning of your code, which can initialize the distributed environments. 

```python
import bmtrain as bmt
bmt.init_distributed(seed=0)
```

## Prepare the model
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

```
config = BertConfig.from_json_file("your/path/to/config.json")
model = Bert(config)
bmt.init_parameters(model)
# bmt.load(model, "your/path/to/pytorch_model.pt")
```

## Perpare the dataset
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

## Train the model
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

        # scale loss to avoid precision underflow of fp16
        loss = optimizer.loss_scale(loss)

        # model backward
        loss.backward()

        # clip gradient norm
        grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale = optimizer.scale, norm_type = 2)

        bmt.optim_step(optimizer, lr_scheduler)

        # print information only on rank 0 when distributed training
        # use bmt.sum_loss(loss) to gather all loss information from all distributed processes
        bmt.print_rank(
            "loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                bmt.sum_loss(loss).item(),
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

## Run your code
You can run the above code using the same launch command as the distributed module of PyTorch.

Choose one of the following commands depending on your version of PyTorch.

* `${MASTER_ADDR}` means the IP address of the master node.
* `${MASTER_PORT}` means the port of the master node.
* `${NNODES}` means the total number of nodes.
* `${GPU_PER_NODE}` means the number of GPUs per node.
* `${NODE_RANK}` means the rank of this node.

#### torch.distributed.launch
```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node ${GPU_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} train.py
```

#### torchrun

```shell
$ torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```


For more information, please refer to the [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility).