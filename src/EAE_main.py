import os
from pathlib import Path
import pdb
import sys
import json
import torch

import numpy as np
from collections import defaultdict

from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OpenEE.backbone.backbone import get_backbone
from OpenEE.input_engineering.EAE_data_processor import (
    TCProcessor,
    SLProcessor,
    Seq2SeqProcessor,
    MRCProcessor
)
from OpenEE.model.model import get_model
from OpenEE.evaluation.metric import (
    compute_F1,
    compute_span_F1,
    compute_seq_F1,
    compute_mrc_F1
)
from OpenEE.evaluation.dump_result import (
    get_leven_submission,
    get_leven_submission_sl,
    get_leven_submission_seq2seq,
    get_maven_submission,
    get_maven_submission_sl,
    get_maven_submission_seq2seq,
    get_duee_submission,
    get_duee_submission_sl,
)
from OpenEE.evaluation.convert_format import (
    get_ace2005_argument_extraction_sl
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
output_dir = Path(
    os.path.join(os.path.join(os.path.join(training_args.output_dir, training_args.task_name), model_args.paradigm),
                 model_name_or_path))
output_dir.mkdir(exist_ok=True, parents=True)
training_args.output_dir = output_dir

# local rank
# training_args.local_rank = int(os.environ["LOCAL_RANK"])

# prepare labels
role2id_path = data_args.role2id_path
data_args.role2id = json.load(open(role2id_path))
model_args.num_labels = len(data_args.role2id)
training_args.label_name = ["labels"]

if model_args.paradigm == "sequence_labeling":
    data_args.role2id = get_bio_labels(data_args.role2id)
    model_args.num_labels = len(data_args.role2id)
data_args.id2role = {id: label for label, id in data_args.role2id.items()}
training_args.id2role = data_args.id2role

# markers 
type2id = json.load(open(data_args.type2id_path))
markers = defaultdict(list)
for label, id in type2id.items():
    markers[label].append(f"<event_{id}>")
    markers[label].append(f"</event_{id}>")
markers["argument"] = ["<argument>", "</argument>"]
data_args.markers = markers
insert_markers = [m for ms in data_args.markers.values() for m in ms]

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
                                           model_args.model_name_or_path, insert_markers, new_tokens=insert_markers)
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
elif model_args.paradigm == "mrc":
    data_class = MRCProcessor
    metric_fn = compute_mrc_F1
    training_args.label_names = ["start_positions", "end_positions"]
else:
    raise ValueError("No such paradigm.")

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file, data_args.train_pred_file, True)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file, data_args.validation_pred_file, False)

# set event types
training_args.data_for_evaluation = eval_dataset.get_data_for_evaluation()

# Trainer 
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metric_fn,
    data_collator=train_dataset.collate_fn,
    tokenizer=tokenizer,
    callbacks=[earlystoppingCallBack]
)
trainer.train()

if training_args.do_predict:
    test_dataset = data_class(data_args, tokenizer, data_args.test_file, data_args.test_pred_file, False)
    training_args.data_for_evaluation = test_dataset.get_data_for_evaluation()
    logits, labels, metrics = trainer.predict(
        test_dataset=test_dataset,
        ignore_keys=["loss"]
    )
    # pdb.set_trace()
    preds = np.argmax(logits, axis=-1)
    if data_args.test_exists_labels:
        # writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
        print(metrics)
        if model_args.paradigm == "sequence_labeling":
            get_ace2005_argument_extraction_sl(preds, labels, data_args.test_file, data_args, test_dataset.is_overflow)
        else:
            pass
    else:
        # save name 
        aggregation = model_args.aggregation
        save_path = os.path.join(training_args.output_dir, f"{model_name_or_path}-{aggregation}.jsonl")
        if model_args.paradigm == "token_classification":
            if data_args.dataset_name == "MAVEN":
                get_maven_submission(preds, test_dataset.get_ids(), save_path)
            elif data_args.dataset_name == "LEVEN":
                get_leven_submission(preds, test_dataset.get_ids(), save_path)
        elif model_args.paradigm == "sequence_labeling":
            if data_args.dataset_name == "MAVEN":
                get_maven_submission_sl(preds, labels, test_dataset.is_overflow, save_path,
                                        json.load(open(role2id_path)), data_args)
            elif data_args.dataset_name == "LEVEN":
                get_leven_submission_sl(preds, labels, test_dataset.is_overflow, save_path,
                                        json.load(open(role2id_path)), data_args)
            elif data_args.dataset_name == "DuEE1.0":
                get_duee_submission_sl(preds, labels, test_dataset.is_overflow, save_path, data_args)
        elif model_args.paradigm == "seq2seq":
            if data_args.dataset_name == "MAVEN":
                get_maven_submission_seq2seq(logits, labels, save_path, json.load(open(role2id_path)), tokenizer,
                                             training_args, data_args)
            elif data_args.dataset_name == "LEVEN":
                get_leven_submission_seq2seq(logits, labels, save_path, json.load(open(role2id_path)), tokenizer,
                                             training_args, data_args)
