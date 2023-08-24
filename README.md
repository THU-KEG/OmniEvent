<div align='center'>
<img src="imgs/Omnievent.png" style="width:350px;">

**A comprehensive, unified and modular event extraction toolkit.**


------

<p align="center">  
    <a href="http://103.238.162.34:9621/">
        <img alt="Demo" src="https://img.shields.io/badge/Demo-site-green">
    </a>
    <a href="https://pypi.org/project/OmniEvent/">
        <img alt="PyPI" src="https://img.shields.io/badge/Pypi-v.0.1.7-blue">
    </a>
    <a href="https://omnievent.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://img.shields.io/badge/Doc-site-red">
    </a>
    <a href="https://github.com/THU-KEG/OmniEvent/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

</div>


# Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [Highlights](#highlights)
- [Installation](#installation)
  - [With pip](#with-pip)
  - [From source](#from-source)
- [Easy Start](#easy-start)
- [Train your Own Model with OmniEvent](#train-your-own-model-with-omnievent)
  - [Step 1: Process the dataset into the unified format](#step-1-process-the-dataset-into-the-unified-format)
  - [Step 2: Set up the customized configurations](#step-2-set-up-the-customized-configurations)
  - [Step 3: Initialize the model and tokenizer](#step-3-initialize-the-model-and-tokenizer)
  - [Step 4: Initialize the dataset and evaluation metric](#step-4-initialize-the-dataset-and-evaluation-metric)
  - [Step 5: Define Trainer and train](#step-5-define-trainer-and-train)
  - [Step 6: Unified Evaluation](#step-6-unified-evaluation)
- [Supported Datasets \& Models \& Contests](#supported-datasets--models--contests)
  - [Datasets](#datasets)
  - [Models](#models)
- [Consistent Evaluation](#consistent-evaluation)
  - [1. Consistent data preprocessing](#1-consistent-data-preprocessing)
  - [2. Output Standardization](#2-output-standardization)
  - [3. Pipeline Evaluation](#3-pipeline-evaluation)
- [Experiments](#experiments)
- [Citation](#citation)


# Overview
OmniEvent is a powerful open-source toolkit for **event extraction**, including **event detection** and **event argument extraction**. We comprehensively cover various paradigms and provide fair and unified evaluations on widely-used **English** and **Chinese** datasets. Modular implementations make OmniEvent highly extensible.

## Highlights
- **Comprehensive Capability**
  - Support to do ***Event Extraction*** at once, and also to independently do its two subtasks: ***Event Detection***, ***Event Argument Extraction***.
  - Cover various paradigms: ***Token Classification***, ***Sequence Labeling***, ***MRC(QA)*** and ***Seq2Seq***.
  - Implement ***Transformer-based*** ([BERT](https://arxiv.org/pdf/1810.04805.pdf), [T5](https://arxiv.org/pdf/1910.10683.pdf), etc.) and ***classical*** ([DMCNN](https://aclanthology.org/P15-1017.pdf), [CRF](http://www.cs.cmu.edu/afs/cs/Web/People/aladdin/papers/pdfs/y2001/crf.pdf), etc.) models.
  - Both ***Chinese*** and ***English*** are supported for all event extraction sub-tasks, paradigms and models. 

- **Unified Benchmark & Evaluation** 
  - Various datasets are processed into a [unified format](https://github.com/THU-KEG/OmniEvent/tree/main/scripts/data_processing#unified-omnievent-format).
  - Predictions of different paradigms are all converted into a [unified candidate set](https://github.com/THU-KEG/OmniEvent/tree/main/OmniEvent/evaluation#convert-the-predictions-of-different-paradigms-to-a-unified-candidate-set) for fair evaluations.
  - Four [evaluation modes](https://github.com/THU-KEG/OmniEvent/tree/main/OmniEvent/evaluation#provide-four-standard-eae-evaluation-modes) (**gold**, **loose**, **default**, **strict**) well cover different previous evaluation settings.

- **Modular Implementation**
  - All models are decomposed into four modules:
    - **Input Engineering**: Prepare inputs and support various input engineering methods like prompting.
    - **Backbone**: Encode text into hidden states.
    - **Aggregation**: Fuse hidden states (e.g., select [CLS], pooling, GCN) to the final event representation. 
    - **Output Head**: Map the event representation to the final outputs, such as Linear, CRF, MRC head, etc. 
  - You can combine and reimplement different modules to design and implement your own new model.

- **Big Model Training & Inference**
  - Efficient training and inference of big event extraction models are supported with [BMTrain](https://github.com/OpenBMB/BMTrain).

- **Easy to Use & Highly Extensible**
  - Open datasets can be downloaded and processed with a single command.
  - Fully compatible with 🤗 [Transformers](https://github.com/huggingface/transformers) and its [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).
  - Users can easily reproduce existing models and build customized models with OmniEvent.



# Installation

## With pip
This repository is tested on Python 3.9+, Pytorch 1.12.1+. OmniEvent can be installed with pip as follows:
```shell
pip install OmniEvent
```

## From source
If you want to install the repository from local source, you can install as follows:
```shell
pip install .
```
And if you want to edit the repositoy, you can 
```shell
pip install -e .
```

# Easy Start
OmniEvent provides several off-the-shelf models for the users. Examples are shown below.

*Make sure you have installed OmniEvent as instructed above. Note that it may take a few minutes to download checkpoint at the first time.*
```python
>>> from OmniEvent.infer import infer

>>> # Even Extraction (EE) Task
>>> text = "2022年北京市举办了冬奥会"
>>> results = infer(text=text, task="EE")
>>> print(results[0]["events"])
[
    {
        "type": "组织行为开幕", "trigger": "举办", "offset": [8, 10],
        "arguments": [
            {   "mention": "2022年", "offset": [9, 16], "role": "时间"},
            {   "mention": "北京市", "offset": [81, 89], "role": "地点"},
            {   "mention": "冬奥会", "offset": [0, 4], "role": "活动名称"},
        ]
    }
]

>>> text = "U.S. and British troops were moving on the strategic southern port city of Basra \ 
Saturday after a massive aerial assault pounded Baghdad at dawn"

>>> # Event Detection (ED) Task
>>> results = infer(text=text, task="ED")
>>> print(results[0]["events"])
[
    { "type": "attack", "trigger": "assault", "offset": [113, 120]},
    { "type": "injure", "trigger": "pounded", "offset": [121, 128]}
]

>>> # Event Argument Extraction (EAE) Task
>>> results = infer(text=text, triggers=[("assault", 113, 120), ("pounded", 121, 128)], task="EAE")
>>> print(results[0]["events"])
[
    {
        "type": "attack", "trigger": "assault", "offset": [113, 120],
        "arguments": [
            {   "mention": "U.S.", "offset": [0, 4], "role": "attacker"},
            {   "mention": "British", "offset": [9, 16], "role": "attacker"},
            {   "mention": "Saturday", "offset": [81, 89], "role": "time"}
        ]
    },
    {
        "type": "injure", "trigger": "pounded", "offset": [121, 128],
        "arguments": [
            {   "mention": "U.S.", "offset": [0, 4], "role": "attacker"},
            {   "mention": "Saturday", "offset": [81, 89], "role": "time"},
            {   "mention": "British", "offset": [9, 16], "role": "attacker"}
        ]
    }
]
```

# Train your Own Model with OmniEvent
OmniEvent can help users easily train and evaluate their customized models on specific datasets.

We show a step-by-step example of using OmniEvent to train and evaluate an ***Event Detection*** model on ***ACE-EN*** dataset in the ***Seq2Seq*** paradigm.
More examples are shown in [examples](./examples).

## Step 1: Process the dataset into the unified format
We provide standard data processing scripts for several commonly-used datasets. Checkout the details in [scripts/data_processing](./scripts/data_processing).
```shell
dataset=ace2005-en  # the dataset name
cd scripts/data_processing/$dataset
bash run.sh
```

## Step 2: Set up the customized configurations
We keep track of the configurations of dataset, model and training parameters via a single `*.yaml` file. See [./configs](./configs) for details.

```python
>>> from OmniEvent.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
>>> from OmniEvent.input_engineering.seq2seq_processor import type_start, type_end

>>> parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
>>> model_args, data_args, training_args = parser.parse_yaml_file(yaml_file="config/all-datasets/ed/s2s/ace-en.yaml")

>>> training_args.output_dir = 'output/ACE2005-EN/ED/seq2seq/t5-base/'
>>> data_args.markers = ["<event>", "</event>", type_start, type_end]
```

## Step 3: Initialize the model and tokenizer
OmniEvent supports various backbones. The users can specify the model and tokenizer in the config file and initialize them as follows.

```python
>>> from OmniEvent.backbone.backbone import get_backbone
>>> from OmniEvent.model.model import get_model

>>> backbone, tokenizer, config = get_backbone(model_type=model_args.model_type, 
                           		       model_name_or_path=model_args.model_name_or_path, 
                           		       tokenizer_name=model_args.model_name_or_path, 
                           		       markers=data_args.markers,
                           		       new_tokens=data_args.markers)
>>> model = get_model(model_args, backbone)
```

## Step 4: Initialize the dataset and evaluation metric
OmniEvent prepares the `DataProcessor` and the corresponding evaluation metrics for different task and paradigms.

***Note that** the metrics here are paradigm-dependent and are **not** used for the final unified evaluation.*

```python
>>> from OmniEvent.input_engineering.seq2seq_processor import EDSeq2SeqProcessor
>>> from OmniEvent.evaluation.metric import compute_seq_F1

>>> train_dataset = EDSeq2SeqProcessor(data_args, tokenizer, data_args.train_file)
>>> eval_dataset = EDSeq2SeqProcessor(data_args, tokenizer, data_args.validation_file)
>>> metric_fn = compute_seq_F1
```

## Step 5: Define Trainer and train
OmniEvent adopts [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) from 🤗 [Transformers](https://github.com/huggingface/transformers) for training and evaluation.

```python
>>> from OmniEvent.trainer_seq2seq import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric_fn,
        data_collator=train_dataset.collate_fn,
        tokenizer=tokenizer,
    )
>>> trainer.train()
```

## Step 6: Unified Evaluation
Since the metrics in Step 4 depend on the paradigm, it is not fair to directly compare the performance of models in different paradigms. 

OmniEvent evaluates models of different paradigms in a unified manner, where the predictions of different models are converted to predictions on the same candidate sets and then evaluated.

```python  
>>> from OmniEvent.evaluation.utils import predict, get_pred_s2s
>>> from OmniEvent.evaluation.convert_format import get_trigger_detection_s2s

>>> logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=EDSeq2SeqProcessor,
                                                    data_args=data_args, data_file=data_args.test_file,
                                                    training_args=training_args)
>>> # paradigm-dependent metrics
>>> print("{} test performance before converting: {}".formate(test_dataset.dataset_name, metrics["test_micro_f1"]))  
ACE2005-EN test performance before converting: 66.4215686224377

>>> preds = get_pred_s2s(logits, tokenizer)
>>> # convert to the unified prediction and evaluate
>>> pred_labels = get_trigger_detection_s2s(preds, labels, data_args.test_file, data_args, None)
ACE2005-EN test performance after converting: 67.41016109045849
```

For those datasets whose test set annotations are not public, such as MAVEN and LEVEN, OmniEvent provide scripts to generate submission files. See [dump_result.py](./OmniEvent/evaluation/dump_result.py) for details.

# Supported Datasets & Models & Contests
Continually updated. Welcome to add more!


## Datasets
<div align='center'>

<table>
    <tr>
        <th>Language</th>
        <th>Domain</th>
        <th>Task</th>  
        <th>Dataset</th>  
    </tr >
    <tr >
        <td rowspan="4">English</td>
        <td>General</td>
        <td>ED</td>
        <td><a href="https://github.com/THU-KEG/MAVEN-dataset"> MAVEN</a></td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED EAE</td>
        <td>ACE-EN</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED EAE</td>
        <td><a href="https://aclanthology.org/D19-1585.pdf">ACE-DYGIE</a> </td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED EAE</td>
        <td>RichERE (KBP+ERE)</td>
    </tr>
    <tr>
        <td rowspan="4">Chinese</td>
        <td>Legal</td>
        <td>ED</td>
        <td><a href="https://github.com/thunlp/LEVEN"> LEVEN </a></td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED EAE</td>
        <td><a href="https://www.luge.ai/#/luge/dataDetail?id=6">DuEE </a></td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED EAE</td>
        <td>ACE-ZH</td>
    </tr>
    <tr>
        <td >Financial</td>
        <td>ED EAE</td>
        <td><a href="https://github.com/TimeBurningFish/FewFC"> FewFC</a></td>


</table>
</div>

## Models

- Paradigm
  - Token Classification (TC)
  - Sequence Labeling (SL)
  - Sequence to Sequence (Seq2Seq)
  - Machine Reading Comprehension (MRC)
- Backbone
  - CNN / LSTM
  - Transformers (BERT, T5, etc.)
- Aggregation
  - Select [CLS]
  - Dynamic/Max Pooling
  - Marker
  - GCN
- Head
  - Linear / CRF / MRC heads

# Consistent Evaluation
OmniEvent provides corresponding remedies for the three discrepancies in event extraction evaluation, as suggested in our ACL 2023 [paper](https://aclanthology.org/2023.findings-acl.586.pdf). 
## 1. Consistent data preprocessing
We provide several preprocessing scripts in `scripts/data_processing`. For ACE 2005, we provide three mainstream scripts: `ace2005-dygie`, `ace2005-oneie`, and `ace2005-en`. Users can easily use the scripts to process the original data into a unified data format. 
## 2. Output Standardization
We implement the output standardization in `OmniEvent/evaluation/convert_format.py`. Specifically, users can use corresponding functions to convert the output of different paradigms into the output space of the token classification method.
## 3. Pipeline Evaluation
As suggested in `OmniEvent/evaluation/README.md`, we provide several evaluation modes for evaluating event argument extraction. We recommend the **strict** mode for comparable evaluation. And we provide a unified extracted trigger set for pipeline evaluation of different event argument extraction methods. The triggers are extracted by an advanced ED model: CLEVE. The extracted triggers for different datasets (ACE 2005, RichERE, and TACKBP 2014-2017) are placed in [here](https://cloud.tsinghua.edu.cn/f/799c8c7c44d24c1db2b7/?dl=1).


# Experiments
We implement and evaluate state-of-the-art methods on some popular benchmarks using OmniEvent, and the results are shown in our ACL 2023 paper "[The Devil is in the Details: On the Pitfalls of Event Extraction Evaluation](https://aclanthology.org/2023.findings-acl.586.pdf)". 

# Citation
If our codes help you, please cite us:
```
@inproceedings{peng2023devil,
  title={The Devil is in the Details: On the Pitfalls of Event Extraction Evaluation},
  author={Peng, Hao and Wang, Xiaozhi and Yao, Feng and Zeng, Kaisheng and Hou, Lei and Li, Juanzi and Liu, Zhiyuan and Shen, Weixing},
  booktitle={Findings of ACL 2023},
  year={2023}
}
```


