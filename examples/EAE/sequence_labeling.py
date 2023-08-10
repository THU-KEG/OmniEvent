import os
from pathlib import Path
import sys
sys.path.append("../../")
import json
import logging

import numpy as np
from collections import defaultdict

from transformers import set_seed, EarlyStoppingCallback

from OmniEvent.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OmniEvent.input_engineering.sequence_labeling_processor import EAESLProcessor

from OmniEvent.model.model import get_model
from OmniEvent.backbone.backbone import get_backbone

from OmniEvent.evaluation.metric import compute_span_F1
from OmniEvent.evaluation.dump_result import get_duee_submission_sl
from OmniEvent.evaluation.convert_format import get_argument_extraction_sl
from OmniEvent.evaluation.utils import predict

from OmniEvent.input_engineering.input_utils import get_bio_labels
from OmniEvent.trainer import Trainer


# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# output dir
model_name_or_path = model_args.model_name_or_path.split("/")[-1]
output_dir = Path(training_args.output_dir, training_args.task_name, model_args.paradigm,
                  f"{model_name_or_path}-{model_args.aggregation}")
output_dir.mkdir(exist_ok=True, parents=True)
training_args.output_dir = str(output_dir)

# logging config 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# prepare labels
data_args.role2id = json.load(open(data_args.role2id_path))
data_args.role2id = get_bio_labels(data_args.role2id)
model_args.num_labels = len(data_args.role2id)

training_args.label_name = ["labels"]

# used for evaluation
training_args.role2id = data_args.role2id 
data_args.id2role = {id: role for role, id in data_args.role2id.items()}

# markers 
type2id = json.load(open(data_args.type2id_path))
markers = defaultdict(list)
for label, id in type2id.items():
    # markers[label].append(f"<event_{id}>")
    # markers[label].append(f"</event_{id}>")
    markers[label].append(f"<event>")
    markers[label].append(f"</event>")
data_args.markers = markers
insert_markers = [m for ms in data_args.markers.values() for m in ms]

# logging
logging.info(data_args)
logging.info(model_args)
logging.info(training_args)

# set seed
set_seed(training_args.seed)

# writter 
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                           model_args.model_name_or_path, insert_markers, model_args,
                                           new_tokens=insert_markers)
model = get_model(model_args, backbone)
model.cuda()

data_class = EAESLProcessor
metric_fn = compute_span_F1

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
    callbacks=[earlystoppingCallBack],
)
if training_args.do_train:
    trainer.train()


if training_args.do_predict:
    for gold, mode in [(True, 'default'), (False, 'default'), (False, 'loose'), (False, 'strict')]:
        data_args.golden_trigger = gold
        data_args.eae_eval_mode = mode

        logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=data_class,
                                                        data_args=data_args, data_file=data_args.test_file,
                                                        training_args=training_args)
        preds = np.argmax(logits, axis=-1)

        logging.info("\n")
        logging.info("{}-EAE Evaluate Mode : {}-{}".format("-" * 25, data_args.eae_eval_mode, "-" * 25))
        logging.info("{}-Use Golden Trigger: {}-{}".format("-" * 25, data_args.golden_trigger, "-" * 25))

        if data_args.test_exists_labels:
            logging.info("{} test performance before converting: {}".format(data_args.dataset_name, metrics))
            get_argument_extraction_sl(preds, labels, data_args.test_file, data_args, test_dataset.is_overflow)
