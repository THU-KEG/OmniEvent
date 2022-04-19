import os 
import pdb 
import sys 
import json 
import torch 

import numpy as np 

from transformers import Trainer, set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OpenEE.input_engineering.data_processor import TCProcessor, SLProcessor
from OpenEE.backbone.backbone import get_backbone
from OpenEE.model.model import ModelForTokenClassification, ModelForSequenceLabeling
from OpenEE.evaluation.metric import compute_accuracy, compute_F1, compute_span_F1
from OpenEE.evaluation.dump_result import get_maven_submission
from OpenEE.input_engineering.input_utils import get_bio_labels

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

# prepare labels
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

model_class = None 
data_class = None  
metric_fn = None 

if model_args.paradigm == "token_classification":
    model_class = ModelForTokenClassification
    data_class = TCProcessor
    metric_fn = compute_F1
elif model_args.paradigm == "sequence_labeling":
    model_class = ModelForSequenceLabeling
    data_class = SLProcessor
    metric_fn = compute_span_F1
else:
    raise ValueError("No such paradigm.")

# model and dataset 
model = model_class(model_args, backbone)
model.cuda()
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
    test_dataset = TCProcessor(data_args, tokenizer, data_args.test_file)
    logits, labels, metrics = trainer.predict(
        test_dataset = test_dataset,
        ignore_keys = ["loss"]
    )
    # pdb.set_trace()
    if data_args.test_exists_labels:
        # writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
        print(metrics)
    else:
        model_name_or_path = model_args.model_name_or_path.split("/")[-1]
        aggregation = model_args.aggregation
        save_name = f"{model_name_or_path}-{aggregation}.jsonl"
        preds = np.argmax(logits, axis=-1)
        get_maven_submission(preds, test_dataset.get_ids(), os.path.join(training_args.output_dir, save_name))


