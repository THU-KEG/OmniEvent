import os
from pathlib import Path
import sys
sys.path.append("../../")
import json
import logging

import numpy as np
from collections import defaultdict

from transformers import set_seed
from transformers import EarlyStoppingCallback

from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OpenEE.backbone.backbone import get_backbone
from OpenEE.input_engineering.mrc_processor import (
    EAEMRCProcessor
)
from OpenEE.model.model import get_model
from OpenEE.evaluation.metric import compute_mrc_F1

from OpenEE.evaluation.utils import (
    predict_eae,
    predict_sub_eae,
)

from OpenEE.trainer import Trainer

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

# logging config 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# prepare labels
role2id_path = data_args.role2id_path
data_args.role2id = json.load(open(role2id_path))
model_args.num_labels = len(data_args.role2id)

# used for evaluation
training_args.role2id = data_args.role2id 
data_args.id2role = {id: role for role, id in data_args.role2id.items()}

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
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                           model_args.model_name_or_path, insert_markers, new_tokens=insert_markers)
model = get_model(model_args, backbone)
model.cuda()

data_class = EAEMRCProcessor
metric_fn = compute_mrc_F1
training_args.label_names = ["argument_left", "argument_right"]

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
    pred_func = predict_sub_eae if data_args.split_infer else predict_eae

    if data_args.test_exists_labels:
        # use gold triggers
        data_args.golden_trigger = True
        logits, labels, metrics, test_dataset = pred_func(trainer, tokenizer, data_class, data_args, training_args)
        print("\n" + "-" * 50 + '\n')
        print("Test File: {}, \nUse_Gold_Trigger, \nMetrics: {}".format(data_args.test_file, metrics))

    for eval_mode in ['default', 'loose', 'strict']:
        print("\n+++++++++++++++++++ Evaluate in [{}] Mode ++++++++++++++++++\n".format(eval_mode))
        data_args.eae_eval_mode = eval_mode
        data_args.golden_trigger = False

        logits, labels, metrics, test_dataset = pred_func(trainer, tokenizer, data_class, data_args, training_args)
        print("\n" + "-" * 50 + '\n')
        print("Test File: {}, \nUse_Pred_Trigger,\nMetrics: {}, ".format(data_args.test_file, metrics))

        # pdb.set_trace()
        preds = np.argmax(logits, axis=-1)
        if data_args.test_exists_labels:
            assert model_args.paradigm == "mrc"
            print("Above is the [{}]test performance for MRC Paradigm.".format(eval_mode))
