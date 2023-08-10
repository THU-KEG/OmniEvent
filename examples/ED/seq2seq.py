import os
from pathlib import Path
import sys
sys.path.append("../../")
import json
import logging

from transformers import set_seed
from transformers import EarlyStoppingCallback

from OmniEvent.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OmniEvent.input_engineering.input_utils import get_plain_label
from OmniEvent.input_engineering.seq2seq_processor import EDSeq2SeqProcessor, type_start, type_end

from OmniEvent.model.model import get_model
from OmniEvent.backbone.backbone import get_backbone

from OmniEvent.evaluation.metric import compute_seq_F1
from OmniEvent.evaluation.utils import predict, dump_preds, get_pred_s2s
from OmniEvent.evaluation.convert_format import get_trigger_detection_s2s
from OmniEvent.evaluation.dump_result import get_leven_submission_seq2seq, get_maven_submission_seq2seq

from OmniEvent.trainer_seq2seq import Seq2SeqTrainer

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) >= 2 and sys.argv[-1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
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
type2id_path = data_args.type2id_path
data_args.type2id = {get_plain_label(k): v for k, v in json.load(open(type2id_path)).items()}
training_args.label_name = ["labels"]

# used for evaluation
training_args.type2id = data_args.type2id
data_args.id2type = {id: type for type, id in data_args.type2id.items()}

# markers
dataset_markers = ["<ace>", "<duee>", "<fewfc>", "<kbp>", "<ere>", "<maven>", "<leven>"]
data_args.markers = ["<event>", "</event>", type_start, type_end] + dataset_markers

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
                                           model_args.model_name_or_path, data_args.markers,
                                           new_tokens=data_args.markers)
model = get_model(model_args, backbone)
model.cuda()

data_class = EDSeq2SeqProcessor
metric_fn = compute_seq_F1

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file)

# Trainer 
trainer = Seq2SeqTrainer(
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
    logits, labels, metrics, test_dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=data_class,
                                                    data_args=data_args, data_file=data_args.test_file,
                                                    training_args=training_args)
    preds = get_pred_s2s(logits, tokenizer)

    logging.info("\n")
    logging.info("{}Predict{}".format("-"*25, "-"*25))

    if data_args.test_exists_labels:
        logging.info("{} test performance: {}".format(data_args.dataset_name, metrics))

        pred_labels = get_trigger_detection_s2s(preds, labels, data_args.test_file, data_args, None)
    else:
        # save name
        aggregation = model_args.aggregation
        save_path = os.path.join(training_args.output_dir, f"{model_name_or_path}-{aggregation}.jsonl")

        if data_args.dataset_name == "MAVEN":
            get_maven_submission_seq2seq(preds, save_path, data_args)
        elif data_args.dataset_name == "LEVEN":
            get_leven_submission_seq2seq(preds, save_path, data_args)
        else:
            pass

        logging.info("{} submission file generated at {}".format(data_args.dataset_name, save_path))

if training_args.do_ED_infer:
    for mode in ["valid", "test"]:
        dump_preds(trainer, tokenizer, data_class, output_dir, model_args, data_args, training_args, mode=mode)
