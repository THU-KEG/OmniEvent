import os 
import pdb 
import json 
from collections import defaultdict
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as span_f1_score
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from .metric import select_start_position
from .dump_result import get_pred_per_mention


def get_ace2005_trigger_detection_sl(preds, labels, data_file, config, is_overflow):
    # get per-word predictions
    preds, labels = select_start_position(preds, labels, False)
    results = []
    label_names = []
    with open(data_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())
            if not is_overflow[i]:
                assert len(preds[i]) == len(item["text"].split())
            candidates = []
            for event in item["events"]:
                for trigger in event["triggers"]:
                    label_names.append(event["type"])
                    candidates.append(trigger)
            for neg_trigger in item["negative_triggers"]:
                label_names.append("NA")
                candidates.append(neg_trigger)
            # loop for converting 
            for candidate in candidates:
                # get word positions
                char_pos = candidate["position"]
                word_pos_start = len(item["text"][:char_pos[0]].split())
                word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                # get predictions
                pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[i], config)
                # record results
                results.append(pred)
            
    pos_labels = list(set(label_names))
    pos_labels.remove("NA")
    micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0
    print("After converting, the micro_f1 is %.4f" % micro_f1)
    return results


def get_ace2005_argument_extraction_sl(preds, labels, data_file, config, is_overflow):
    # get per-word predictions
    preds, labels = select_start_position(preds, labels, False)
    results = []
    label_names = []
    with open(data_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())
            if not is_overflow[i]:
                assert len(preds[i]) == len(item["text"].split())
            candidates = []
            positive_mentions = []
            for event in item["events"]:
                for trigger in event["triggers"]:
                    for argument in trigger["arguments"]:
                        for mention in argument["mentions"]:
                            label_names.append(argument["role"])
                            candidates.append(mention)
                            positive_mentions.append(mention["mention_id"])
            for neg_trigger in item["negative_triggers"]:
                label_names.append("NA")
                candidates.append(neg_trigger)
            # loop for converting 
            for candidate in candidates:
                # get word positions
                char_pos = candidate["position"]
                word_pos_start = len(item["text"][:char_pos[0]].split())
                word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                # get predictions
                pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[i], config)
                # record results
                results.append(pred)
            
    pos_labels = list(set(label_names))
    pos_labels.remove("NA")
    micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0
    print("After converting, the micro_f1 is %.4f" % micro_f1)
    return results
