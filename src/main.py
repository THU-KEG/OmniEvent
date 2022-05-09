import os
from pathlib import Path 
import pdb 
import sys 
import json 
import torch 

import numpy as np 

from transformers import set_seed 
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OpenEE.backbone.backbone import get_backbone
from OpenEE.input_engineering.data_processor import (
    TCProcessor,
    SLProcessor,
    Seq2SeqProcessor
)
from OpenEE.model.model import get_model
from OpenEE.evaluation.metric import (
    compute_F1, 
    compute_span_F1, 
    compute_seq_F1
)
from OpenEE.evaluation.dump_result import (
    get_leven_submission,
    get_leven_submission_sl,
    get_leven_submission_seq2seq,
    get_maven_submission, 
    get_maven_submission_sl,
    get_maven_submission_seq2seq
)
from OpenEE.input_engineering.input_utils import get_bio_labels
from OpenEE.trainer import Trainer
from OpenEE.trainer_seq2seq import Seq2SeqTrainer

# from torch.utils.tensorboard import SummaryWriter

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# output dir
model_name_or_path = model_args.model_name_or_path.split("/")[-1]
output_dir = Path(os.path.join(os.path.join(training_args.output_dir, model_args.paradigm), model_name_or_path))
output_dir.mkdir(exist_ok=True, parents=True)
training_args.output_dir = output_dir

# local rank
# training_args.local_rank = int(os.environ["LOCAL_RANK"])

# prepare labels
label2id_path = data_args.label2id
data_args.label2id = json.load(open(data_args.label2id))
model_args.num_labels = len(data_args.label2id)
training_args.label_name = ["labels"]

if model_args.paradigm == "sequence_labeling":
    data_args.label2id = get_bio_labels(data_args.label2id)
    model_args.num_labels = len(data_args.label2id)

# markers 
# data_args.markers =  ["[unused0]", "[unused1]"]
data_args.markers =  ["<event>", "</event>"]

print(data_args, model_args, training_args)

# set seed
set_seed(training_args.seed)

# writter 
# writer = SummaryWriter(training_args.output_dir)
# tensorboardCallBack = TensorBoardCallback(writer)
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience, \
                                                early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path, \
                                        model_args.model_name_or_path, data_args.markers, new_tokens=data_args.markers)
model = get_model(model_args, backbone)
model.cuda()
data_class = None  
metric_fn = None 

if model_args.paradigm == "token_classification":
    data_class = TCProcessor
    metric_fn = compute_F1
elif model_args.paradigm == "sequence_labeling":
    data_class = SLProcessor
    metric_fn = compute_span_F1
elif model_args.paradigm == "seq2seq":
    data_class = Seq2SeqProcessor
    metric_fn = compute_seq_F1
else:
    raise ValueError("No such paradigm.")

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file)

# Trainer 
trainer = Trainer(
    args = training_args,
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    compute_metrics = metric_fn,
    data_collator = train_dataset.collate_fn,
    tokenizer = tokenizer,
    callbacks = [earlystoppingCallBack]
)
trainer.train()


if training_args.do_predict:
    test_dataset = data_class(data_args, tokenizer, data_args.test_file)
    logits, labels, metrics = trainer.predict(
        test_dataset = test_dataset,
        ignore_keys = ["loss"]
    )
    # pdb.set_trace()
    if data_args.test_exists_labels:
        # writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
        print(metrics)
    else:
        # save name 
        aggregation = model_args.aggregation
        save_path = os.path.join(training_args.output_dir, f"{model_name_or_path}-{aggregation}.jsonl")
        preds = np.argmax(logits, axis=-1)
        if model_args.paradigm == "token_classification":
            if data_args.dataset_name == "MAVEN":
                get_maven_submission(preds, test_dataset.get_ids(), save_path)
            elif data_args.dataset_name == "LEVEN":
                get_leven_submission(preds, test_dataset.get_ids(), save_path)
        elif model_args.paradigm == "sequence_labeling":
            if data_args.dataset_name == "MAVEN":
                get_maven_submission_sl(preds, labels, test_dataset.is_overflow, save_path, json.load(open(label2id_path)), data_args)
            elif data_args.dataset_name == "LEVEN":
                get_leven_submission_sl(preds, labels, test_dataset.is_overflow, save_path, json.load(open(label2id_path)), data_args)
        elif model_args.paradigm == "seq2seq":
            if data_args.dataset_name == "MAVEN":
                get_maven_submission_seq2seq(logits, labels, save_path, json.load(open(label2id_path)), tokenizer, training_args, data_args)
            elif data_args.dataset_name == "LEVEN":
                get_leven_submission_seq2seq(logits, labels, save_path, json.load(open(label2id_path)), tokenizer, training_args, data_args)



