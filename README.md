# OmniEvent

## Overview
OmniEvent is a powerful Event Extraction toolkit that provides comprehensive implementations of state-of-the-art methods 
along with unified workflows of data processing and model evaluation. 

### Features
- **Comprehensive Implementations**
  - All sub-tasks, Event Detection, Event Argument Extraction and Event Extraction, are considered.
  - Various paradigms, [Token Classification](), Sequence Labeling, [MRC(QA)]() and [Seq2Seq](), are deployed.
  - Transformers-based ([BERT](), [T5](), etc.) and classical models (CNN, LSTM, CRF, etc.) are implemented.
  - Both Chinese and English are supported for all event extraction sub-tasks, paradigms and models. 
- **Unified Benchmark & Evaluation** 
  - Different datasets for event detection and extraction are processed into a [unified format]().
  - Predicted results of different paradigms are all converted into word level for comparable evaluation.
- **Support Big Model Training & Inference**
  - Efficient training and inference of big models for event extraction are supported with [BMTrain](https://github.com/OpenBMB/BMTrain).
- **Easy to Use and Highly Extensible**
  - Datasets can be downloaded (if open-sourced) and processed with a single command.
  - OmniEvent is fully compatible with 🤗 [Transformers](https://github.com/huggingface/transformers) and adopts [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) for training and evaluation.
  - Users can adopt existing models directly or adapt OmniEvent to build customized models at will.  



## Installation

## Easy Start
Make sure you have installed OmniEvent as instructed above. Then import our package and load pre-trained models.
Note that it may take a few minutes to download checkpoint for the first time. 

```python
>>> from OpenEE.infer import infer
>>> infer(task="ED", text="U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn")
>>> infer(task="EAE", text="U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn")
>>> infer(task="EE", text="U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn")
```

## Use OmniEvent

### Step1: Process the data into the unified format
```shell
cd scripts/data_processing/maven
python maven.py
```

### Step2: Set up the training configurations

```python
>>> from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
>>> from OpenEE.input_engineering.seq2seq_processor import type_start, type_end
>>> parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
>>> model_args, data_args, training_args = parser.parse_yaml_file(yaml_file="config/ed/s2s/maven.yaml")
>>> training_args.output_dir = 'output/MAVEN/ED/seq2seq/mt5-base/'
>>> data_args.markers = ["<event>", "</event>", type_start, type_end]
```

### Step3: Define the backbone

```python
>>> from OpenEE.backbone.backbone import get_backbone
>>> from OpenEE.model.model import get_model
>>> backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                               model_args.model_name_or_path, data_args.markers,
                                               new_tokens=data_args.markers)
>>> model = get_model(model_args, backbone)
>>> model.cuda()
```

### Step4: Initialize dataset and evaluation metric

```python
>>> from OpenEE.input_engineering.seq2seq_processor import EDSeq2SeqProcessor
>>> from OpenEE.evaluation.metric import compute_seq_F1
>>> train_dataset = data_class(data_args, tokenizer, data_args.train_file)
>>> eval_dataset = data_class(data_args, tokenizer, data_args.validation_file)
>>> metric_fn = compute_seq_F1
```
### Step5: Define Trainer and train

```python
>>> from OpenEE.trainer_seq2seq import Seq2SeqTrainer
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

### Step6: Unified Evaluation
```python
>>> from OpenEE.evaluation.utils import predict, get_pred_s2s
>>> from OpenEE.evaluation.convert_format import get_ace2005_trigger_detection_s2s
>>> logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=data_class,
                                                    data_args=data_args, data_file=data_args.test_file,
                                                    training_args=training_args)
>>> preds = get_pred_s2s(logits, tokenizer)
>>> print("test performance before converting: {}".format(metrics))
>>> # convert to word-level prediction and evaluate
>>> pred_labels = get_ace2005_trigger_detection_s2s(preds, labels, data_args.test_file, data_args, None)
```

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
	    <td>ACE-ONEID</td>
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