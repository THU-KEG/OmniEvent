<div align='center'>
<img src="imgs/Omnievent.png" style="width:350px;">

**A comprehensive, unified and modular event extraction toolkit.**


------

<p align="center">  
    <a href="placeholder">
        <img alt="Demo" src="https://img.shields.io/badge/Demo-site-green">
    </a>
    <a href="placeholder">
        <img alt="PyPI" src="https://img.shields.io/badge/Pypi-v.0.1.0-blue">
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
- [Easy Start](#easy-start)
- [Train your Own Model with OmniEvent](#train-your-own-model-with-omnievent)
  - [Step 1: Process the dataset into the unified format](#step-1-process-the-dataset-into-the-unified-format)
  - [Step 2: Set up the customized configurations](#step-2-set-up-the-customized-configurations)
  - [Step 3: Initialize the model and tokenizer](#step-3-initialize-the-model-and-tokenizer)
  - [Step 4: Initialize the dataset and evaluation metric](#step-4-initialize-the-dataset-and-evaluation-metric)
  - [Step 5: Define Trainer and train](#step-5-define-trainer-and-train)
  - [Step 6: Unified Evaluation](#step-6-unified-evaluation)
- [Supported Datasets & Models](#supported-datasets--models)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Contests](#contests)
- [Experiments](#experiments)


# Overview
OmniEvent is a powerful open-source toolkit for **event extraction**, including **event detection** and **event argument extraction**. We comprehensively cover various paradigms and provide fair and unified evaluations on widely-used **English** and **Chinese** datasets. Modular implementations make OmniEvent highly extensible.

## Highlights
- **Comprehensive Capability**
  - Support to do ***Event Extraction*** at once, and also to independently do its two subtasks: ***Event Detection***, ***Event Argument Extraction***.
  - Cover various paradigms: ***Token Classification***, ***Sequence Labeling***, ***MRC(QA)*** and ***Seq2Seq***.
  - Implement ***Transformer-based*** ([BERT](https://arxiv.org/pdf/1810.04805.pdf), [T5](https://arxiv.org/pdf/1910.10683.pdf), etc.) and ***classical*** ([DMCNN](https://aclanthology.org/P15-1017.pdf), [CRF](http://www.cs.cmu.edu/afs/cs/Web/People/aladdin/papers/pdfs/y2001/crf.pdf), etc.) models.
  - Both ***Chinese*** and ***English*** are supported for all event extraction sub-tasks, paradigms and models. 

- **Modular Implementation**
  - All models are decomposed into four modules:
    - **Input Engineering**: Prepare inputs and support various input engineering methods like prompting.
    - **Backbone**: Encode text into hidden states.
    - **Aggregation**: Fuse hidden states (e.g., select [CLS], pooling, GCN) to the final event representation. 
    - **Output Head**: Map the event representation to the final outputs, such as Linear, CRF, MRC head, etc. 
  - You can combine and reimplement different modules to design and implement your own new model.

- **Unified Benchmark & Evaluation** 
  - Various datasets are processed into a [unified format](https://github.com/THU-KEG/OmniEvent/tree/main/scripts/data_processing#unified-omnievent-format).
  - Predictions of different paradigms are all converted into a [unified candidate set](https://github.com/THU-KEG/OmniEvent/tree/main/OmniEvent/evaluation#convert-the-predictions-of-different-paradigms-to-a-unified-candidate-set) for fair evaluations.
  - Four [evaluation modes](https://github.com/THU-KEG/OmniEvent/tree/main/OmniEvent/evaluation#provide-four-standard-eae-evaluation-modes) (**gold**, **loose**, **default**, **strict**) well cover different previous evaluation settings.

- **Big Model Training & Inference**
  - Efficient training and inference of big event extraction models are supported with [BMTrain](https://github.com/OpenBMB/BMTrain).

- **Easy to Use & Highly Extensible**
  - Open datasets can be downloaded and processed with a single command.
  - Fully compatible with 🤗 [Transformers](https://github.com/huggingface/transformers) and its [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).
  - Users can easily reproduce existing models and build customized models with OmniEvent.



# Installation




# Easy Start
OmniEvent provides several off-the-shelf models for the users. Examples are shown below.

*Make sure you have installed OmniEvent as instructed above. Note that it may take a few minutes to download checkpoint at the first time.*
```python
>>> from OmniEvent.infer import infer

>>> # Even Extraction (EE) Task
>>> text = "2022年北京市举办了冬奥会"
>>> infer(text=text, task="EE")
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
>>> infer(text=text, triggers=[("assault", 113, 120), ("pounded", 121, 128)], task="EAE")
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
>>> from OmniEvent.evaluation.convert_format import get_ace2005_trigger_detection_s2s

>>> logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=data_class,
                                                    data_args=data_args, data_file=data_args.test_file,
                                                    training_args=training_args)
>>> # paradigm-dependent metrics
>>> print("{} test performance before converting: {}".formate(test_dataset.dataset_name, metrics["test_micro_f1"]))  
ACE2005-EN test performance before converting: 66.4215686224377

>>> preds = get_pred_s2s(logits, tokenizer)
>>> # convert to the unified prediction and evaluate
>>> pred_labels = get_ace2005_trigger_detection_s2s(preds, labels, data_args.test_file, data_args, None)
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
'

## Contests
OmniEvent plans to support various event extraction contest. Currently, we support the following contests and the list is continually updated!

- [MAVEN Event Detection Challenge](https://codalab.lisn.upsaclay.fr/competitions/395)
- [CAIL 2022: Event Detection Track](http://cail.cipsc.org.cn/task1.html?raceID=1&cail_tag=2022)
- [LUGE: Information Extraction Track](https://aistudio.baidu.com/aistudio/competition/detail/46/0/task-definition)

# Experiments
We implement and evaluate state-of-the-art methods on some popular benchmarks using OmniEvent. The results of all Event Detection experiments are shown in the table below. The full results can be acessed via the links below.

- [Experiments of base models on <u>**All ED Benchmars**</u>](https://docs.qq.com/sheet/DRW5QQU1tZ2ViZlFo?tab=qp276f)
- [Experiments of base models on <u>**All EAE Benchmarks**</u>](https://docs.qq.com/sheet/DRW5QQU1tZ2ViZlFo?tab=b0zjme)
- [Experiments of <u>**All ED Models**</u> on ACE-EN+](https://docs.qq.com/sheet/DRW5QQU1tZ2ViZlFo?tab=odcgnh)
- [Experiments of <u>**All EAE Models**</u> on ACE-EN+](https://docs.qq.com/sheet/DRW5QQU1tZ2ViZlFo?tab=jxc1ea)


<div align='center'>

<table>
    <tr>
        <td rowspan="2" align="center">Language</td>
        <td rowspan="2" align="center">Domain</td>
        <td rowspan="2" align="center">Benchmark</td>
        <td rowspan="2" align="center">Paradigm</td>
		<td colspan="2" align="center">Dev F1-score</td>
        <td colspan="2" align="center">Test F1-score</td>
    </tr>
    <tr>
        <td align="center">Paradigm-based</td>
        <td align="center">Unified</td>
        <td align="center">Paradigm-based</td>
        <td align="center">Unified</td>
    </tr>
    <tr>
        <td rowspan="12" align="center">English</td>
        <td rowspan="3" align="center">General</td>
        <td rowspan="3" align="center">MAVEN</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">68.80 </td>
        <td align="center">--</td>
        <td align="center">68.64 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">66.75 </td>
        <td align="center">67.90 </td>
        <td align="center">--</td>
        <td align="center">68.64 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">61.23 </td>
        <td align="center">61.86 </td>
        <td align="center">--</td>
        <td align="center">61.86 </td>
    </tr>
    <tr>
        <td rowspan="3" align="center">General</td>
        <td rowspan="3" align="center">ACE-EN</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">80.47 </td>
        <td align="center">--</td>
        <td align="center">74.13 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">77.72 </td>
        <td align="center">79.44 </td>
        <td align="center">74.86 </td>
        <td align="center">75.63 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">75.88 </td>
        <td align="center">76.73 </td>
        <td align="center">73.09 </td>
        <td align="center">72.97 </td>
    </tr>
    <tr>
        <td rowspan="3" align="center">General</td>
        <td rowspan="3" align="center">ACE-dygie</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">73.61 </td>
        <td align="center">--</td>
        <td align="center">68.63 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">71.58 </td>
        <td align="center">71.75 </td>
        <td align="center">68.63 </td>
        <td align="center">68.63 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">71.61 </td>
        <td align="center">72.08 </td>
        <td align="center">65.41 </td>
        <td align="center">65.99 </td>
    </tr>
    <tr>
        <td rowspan="3" align="center">General</td>
        <td rowspan="3" align="center">RichERE</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">68.75 </td>
        <td align="center">--</td>
        <td align="center">51.43 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">68.46 </td>
        <td align="center">66.05 </td>
        <td align="center">50.13 </td>
        <td align="center">50.77 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">63.21 </td>
        <td align="center">62.74 </td>
        <td align="center">50.07 </td>
        <td align="center">51.35 </td>
    </tr>
    <tr>
        <td rowspan="12" align="center">Chinese</td>
        <td rowspan="3" align="center">General</td>
        <td rowspan="3" align="center">ACE-ZH</td>
        <td align="center" >TC</td>
        <td align="center">--</td>
        <td align="center">79.76 </td>
        <td align="center">--</td>
        <td align="center">75.77 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">75.41 </td>
        <td align="center">75.88 </td>
        <td align="center">72.23 </td>
        <td align="center">75.93 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">69.45 </td>
        <td align="center">73.17 </td>
        <td align="center">63.37 </td>
        <td align="center">71.61 </td>
    </tr>
    <tr>
        <td rowspan="3" align="center">General</td>
        <td rowspan="3" align="center">DuEE</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">92.20 </td>
        <td align="center">--</td>
        <td align="center">--</td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">85.95 </td>
        <td align="center">89.62 </td>
        <td align="center">--</td>
        <td align="center">--</td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">81.61 </td>
        <td align="center">85.85 </td>
        <td align="center">--</td>
        <td align="center">--</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">Legal</td>
        <td rowspan="3" align="center">LEVEN</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">85.18 </td>
        <td align="center">--</td>
        <td align="center">85.23 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">81.09 </td>
        <td align="center">84.16 </td>
        <td align="center">--</td>
        <td align="center">84.66 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">78.14 </td>
        <td align="center">81.29 </td>
        <td align="center">--</td>
        <td align="center">81.41 </td>
    </tr>
    <tr>
        <td rowspan="3" align="center">Financial</td>
        <td rowspan="3" align="center">FewFC</td>
        <td align="center">TC</td>
        <td align="center">--</td>
        <td align="center">69.28 </td>
        <td align="center">--</td>
        <td align="center">67.15 </td>
    </tr>
    <tr>
        <td align="center">SL</td>
        <td align="center">71.13 </td>
        <td align="center">63.75 </td>
        <td align="center">68.99 </td>
        <td align="center">62.31 </td>
    </tr>
    <tr>
        <td align="center">S2S</td>
        <td align="center">69.89 </td>
        <td align="center">74.46 </td>
        <td align="center">69.16 </td>
        <td align="center">71.33 </td>
    </tr>
</table>
</div>

