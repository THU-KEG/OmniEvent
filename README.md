<div align='center'><img src="imgs/Omnievent.png" style="width:256px;"></div>



## Overview
OmniEvent is a powerful event extraction toolkit that provides comprehensive and compact implementations of various event extraction methods. OmniEvent benchmarks existing main-stream event extraction paradigms (token classification, sequence labeling, seq2seq, and machine reading comprehension) on commenly-used English and Chinese datasets. Meanwhile, OmniEvent provides unified evaluation methods and thus a fair comparsion of different models. 

### Important Features
- **Comprehensive Implementations**
  - All sub-tasks, ***Event Detection***, ***Event Argument Extraction*** and ***Event Extraction***, are considered.
  - Various paradigms, ***Token Classification***, ***Sequence Labeling***, ***MRC(QA)*** and ***Seq2Seq***, are deployed.
  - ***Transformers-based*** ([BERT](), [T5](), etc.) and ***classical*** models (CNN, LSTM, CRF, etc.) are implemented.
  - Both Chinese and English are supported for all event extraction sub-tasks, paradigms and models. 
- **Modular Implementation**
  - All methods are decomposed into four parts
    - **Input Engineering**: Prepare inputs and support various input manipulation, such as prompt.
    - **backbone**: Encode text into hidden states.
    - **aggregation**: Aggragate hidden states (e.g., select [CLS], pooling, GCN) for the final event representation. 
    - **head**: Map the event representation to the final outputs, such as classification head, CRF. 

  - One can combine different modules to design a new model

- **Unified Benchmark & Evaluation** 
  - Different datasets for event detection and extraction are processed into a [unified format]().
  - Predicted results of different paradigms are all converted into word level for comparable evaluation.
  - Design three evaluation modes (**loose**, **default**, **strict**) for a fair comparison of different methods.
- **Support Big Model Training & Inference**
  - Efficient training and inference of big models for event extraction are supported with [BMTrain](https://github.com/OpenBMB/BMTrain).
- **Easy to Use & Highly Extensible**
  - Datasets can be downloaded (if open-sourced) and processed with a single command.
  - OmniEvent is fully compatible with 🤗 [Transformers](https://github.com/huggingface/transformers) and adopts [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) for training and evaluation.
  - Users can adopt existing models directly or adapt OmniEvent to build customized models at will.  



## Installation




## Easy Start
OmniEvent provides ready-to-use models for the users. Examples are shown below. 

*Make sure you have installed OmniEvent as instructed above. Note that it may take a few minutes to download checkpoint for the first time.*
```python
>>> from OmniEvent.infer import infer

>>> text = "U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn"

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
    { "type": "attack", "trigger": "assault", "offset": [113, 120], "arguments": [{"mention": "U.S.", "offset": [0, 4], "role": "attacker"}, {"mention": "British", "offset": [9, 16], "role": "attacker"}, {"mention": "Saturday", "offset": [81, 89], "role": "time"}]
    { "type": "injure", "trigger": "pounded", "offset": [121, 128], "arguments": [{"mention": "U.S.", "offset": [0, 4], "role": "attacker"}, {"mention": "Saturday", "offset": [81, 89], "role": "time"}, {"mention": "British", "offset": [9, 16], "role": "attacker"}]}
]

>>> # Even Extraction (EE) Task
>>> infer(text=text, task="EE")
>>> print(results[0]["events"])
[
    { "type": "attack", "trigger": "assault", "offset": [113, 120], "arguments": [{"mention": "U.S.", "offset": [0, 4], "role": "attacker"}, {"mention": "British", "offset": [9, 16], "role": "attacker"}, {"mention": "Saturday", "offset": [81, 89], "role": "time"}]
    { "type": "injure", "trigger": "pounded", "offset": [121, 128], "arguments": [{"mention": "U.S.", "offset": [0, 4], "role": "attacker"}, {"mention": "Saturday", "offset": [81, 89], "role": "time"}, {"mention": "British", "offset": [9, 16], "role": "attacker"}]}
]
```

## Customized Use of OmniEvent
OmniEvent can help users easily train and evaluate their customized models on a specific dataset. 

We show a step-by-step example of using OmniEvent to train and evlauate an ***Event Detection*** model on ***ACE-EN*** dataset in the ***Seq2Seq*** paradigm.
### Step 1: Process the dataset into the unified format
We provide standard data processing scripts for commonly-adopted datasets. Checkout the details in [scripts/data_processing](./scripts/data_processing).
```shell
dataset=ace2005-en  # the dataset name
cd scripts/data_processing/$dataset
bash run.sh
```

### Step 2: Set up the customized configurations
We keep track of the configurations of dataset, model and training parameters via a single *.yaml file. See [./configs](./configs) for details.

```python
>>> from OmniEvent.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
>>> from OmniEvent.input_engineering.seq2seq_processor import type_start, type_end

>>> parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
>>> model_args, data_args, training_args = parser.parse_yaml_file(yaml_file="config/ed/s2s/ace2005-en.yaml")

>>> training_args.output_dir = 'output/ACE2005-EN/ED/seq2seq/mt5-base/'
>>> data_args.markers = ["<event>", "</event>", type_start, type_end]
```

### Step 3: Initialize the model and tokenizer
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
>>> model.cuda()
```

### Step 4: Initialize dataset and evaluation metric
OmniEvent prepares the DataProcessor and the corresponding evaluation metrics for different task and paradigms.

***Note that** the metrics here are paradigm-dependent and are **not** used for the final unified evaluation.*

```python
>>> from OmniEvent.input_engineering.seq2seq_processor import EDSeq2SeqProcessor
>>> from OmniEvent.evaluation.metric import compute_seq_F1

>>> train_dataset = data_class(data_args, tokenizer, data_args.train_file)
>>> eval_dataset = data_class(data_args, tokenizer, data_args.validation_file)
>>> metric_fn = compute_seq_F1
```
### Step 5: Define Trainer and train
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

### Step 6: Unified Evaluation
Since the metrics in Step 4 depend on the paradigm, it is not fair to directly compare the performance of different paradigms. 

OmniEvent evaluates models of different paradigms in a unifed manner, where the predictions of different models are converted to word-level and then evaluated.
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
>>> # convert to word-level prediction and evaluate
>>> pred_labels = get_ace2005_trigger_detection_s2s(preds, labels, data_args.test_file, data_args, None)
ACE2005-EN test performance after converting: 67.41016109045849
```
For those datasets whose test set annotations are not given, such as MAVEN and LEVEN, OmniEvent provide APIs to generate submission files. See [dump_result.py](./OmniEvent/evaluation/dump_result.py) for details.

## Datasets & Models Support


Datasets
<table>
    <tr>
        <th>Language</th>
        <th>Domain</th>
        <th>Task</th>  
        <th>Dataset</th>  
    </tr >
    <tr >
        <td rowspan="6">English</td>
        <td>General</td>
        <td>ED</td>
        <td>MAVEN</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED&EAE</td>
        <td>ACE-EN</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED&EAE</td>
        <td>ACE-DYGIE</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED&EAE</td>
        <td>ACE-ONEIE</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED&EAE</td>
        <td>KBP</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED&EAE</td>
        <td>ERE</td>
    </tr>
    <tr>
        <td rowspan="4">Chinese</td>
        <td>General</td>
        <td>ED&EAE</td>
        <td>ACE-ZH</td>
    </tr>
    <tr>
        <td>General</td>
        <td>ED&EAE</td>
        <td>DuEE</td>
    </tr>
    <tr>
        <td >Legal</td>
        <td>ED</td>
        <td>LEVEN</td>
    </tr>
    <tr>
        <td >Financial</td>
        <td>ED&EAE</td>
        <td>FewFC</td>


</table>

Models
<table>
    <tr>
        <th>Paradigm</th>
        <th>Backbone</th>
        <th>Aggregation / Head</th>  
    </tr >
    <tr >
        <td>Token Classification </td>
        <td>CNN <br> LSTM <br> BERT <br> RoBERTa </td>
        <td>CLS <br> Dynamic Pooling <br> Marker <br> Max Pooling</td>
    </tr>
    <tr >
        <td>Sequence Labeling </td>
        <td>CNN <br> LSTM <br> BERT <br> RoBERTa </td>
        <td> CRF Head <br> Classification Head </td>
    </tr>
    <tr >
        <td>Seq2Seq </td>
        <td>T5 <br> MT5 </td>
        <td> / </td>
    </tr>
    <tr >
        <td>MRC </td>
        <td>LSTM <br> BERT <br> RoBERTa </td>
        <td> Classification Head </td>
    </tr>


</table>