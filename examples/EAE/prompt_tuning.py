import copy
import os
from pathlib import Path
import pdb
import sys
sys.path.append("../../")
import json
import torch
import logging

import numpy as np
from tqdm import tqdm
from collections import defaultdict

from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback
from opendelta import SoftPromptModel

from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OpenEE.backbone.backbone import get_backbone
from OpenEE.input_engineering.seq2seq_processor import (
    EAESeq2SeqProcessor
)
from OpenEE.model.model import get_model
from OpenEE.evaluation.metric import (
    compute_seq_F1,
)

from OpenEE.evaluation.utils import (
    predict_eae,
    predict_sub_eae,
)

from OpenEE.input_engineering.input_utils import get_bio_labels
from OpenEE.trainer_seq2seq import Seq2SeqTrainer, ConstrainedSeq2SeqTrainer

# from torch.utils.tensorboard import SummaryWriter

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) >= 2 and sys.argv[1].endswith(".yaml"):
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

# logging config 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# prepare labels
role2id_path = data_args.role2id_path
data_args.role2id = json.load(open(role2id_path))
all_roles_except_na = copy.deepcopy(list(data_args.role2id.keys()))
all_roles_except_na.remove("NA")

# markers 
type2id = json.load(open(data_args.type2id_path))
markers = defaultdict(list)
for label, id in type2id.items():
    markers[label].append(f"<event>")
    markers[label].append(f"</event>")
data_args.markers = markers
insert_markers = [m for ms in data_args.markers.values() for m in ms]
insert_markers.append("[SEP]")
for i in range(10):
    insert_markers.append(f"<extra_id_{i}>")
print(data_args, model_args, training_args)

# set seed
set_seed(training_args.seed)

# writter 
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience, \
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path, \
                                           model_args.model_name_or_path, insert_markers, new_tokens=insert_markers)
delta_model = SoftPromptModel(backbone_model=backbone)
delta_model.freeze_module(set_state_dict=True)
model = get_model(model_args, backbone)
model.cuda()

data_class = EAESeq2SeqProcessor
metric_fn = compute_seq_F1

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file, data_args.train_pred_file, True)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file, data_args.validation_pred_file, False)

# set event types
training_args.data_for_evaluation = eval_dataset.get_data_for_evaluation()

# Trainer 
trainer = ConstrainedSeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metric_fn,
    data_collator=train_dataset.collate_fn,
    tokenizer=tokenizer,
    callbacks=[earlystoppingCallBack],
    decoding_type_schema={"role_list": all_roles_except_na}
)
trainer.train()


if training_args.do_predict:
    if not data_args.split_infer:
        logits, labels, metrics, test_dataset = predict_eae(trainer, tokenizer, data_class, data_args, training_args)
    else:
        logits, labels, test_dataset = predict_sub_eae(trainer, tokenizer, data_class, data_args, training_args)

    # pdb.set_trace()
    preds = np.argmax(logits, axis=-1)
    if data_args.test_exists_labels:
        # writer.add_scalar(tag="test_accuracy", scalar_value=metrics["test_accuracy"])
        print(metrics)
    else:
        pass 