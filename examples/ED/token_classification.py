import os
from pathlib import Path
import sys
sys.path.append("../../")
import json
import logging
import numpy as np

from transformers import set_seed
from transformers import EarlyStoppingCallback

from OpenEE.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OpenEE.backbone.backbone import get_backbone
from OpenEE.input_engineering.token_classification_processor import EDTCProcessor

from OpenEE.model.model import get_model

from OpenEE.evaluation.metric import compute_F1
from OpenEE.evaluation.utils import predict, dump_preds
from OpenEE.evaluation.dump_result import get_leven_submission, get_maven_submission

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
                 f"{model_name_or_path}-{model_args.aggregation}"))
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
type2id_path = data_args.type2id_path
data_args.type2id = json.load(open(type2id_path))
model_args.num_labels = len(data_args.type2id)
training_args.label_name = ["labels"]

# used for evaluation
training_args.type2id = data_args.type2id
data_args.id2type = {id: type for type, id in data_args.type2id.items()}

# markers
data_args.markers = ["<event>", "</event>"]

print(data_args, model_args, training_args)

# set seed
set_seed(training_args.seed)

# writter 
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                           model_args.model_name_or_path, data_args.markers, model_args,
                                           new_tokens=data_args.markers)
model = get_model(model_args, backbone)
model.cuda()
data_class = EDTCProcessor
metric_fn = compute_F1

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file)

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
    logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=data_class,
                                                    data_args=data_args, data_file=data_args.test_file,
                                                    training_args=training_args)
    if data_args.test_exists_labels:
        print(metrics)
    else:
        # save name 
        aggregation = model_args.aggregation
        save_path = os.path.join(training_args.output_dir, f"{model_name_or_path}-{aggregation}.jsonl")
        preds = np.argmax(logits, axis=-1)

        if data_args.dataset_name == "MAVEN":
            get_maven_submission(preds, test_dataset.get_ids(), save_path)
        elif data_args.dataset_name == "LEVEN":
            get_leven_submission(preds, test_dataset.get_ids(), save_path)
        else:
            pass


if training_args.do_ED_infer:
    for mode in ["train", "valid", "test"]:
        dump_preds(trainer, tokenizer, data_class, output_dir, model_args, data_args, training_args, mode=mode)