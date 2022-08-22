import logging
import os 
import pdb 
import json 
from collections import defaultdict
from typing import List

from sklearn.metrics import f1_score

from .metric import select_start_position, compute_unified_micro_f1
from .dump_result import get_pred_per_mention
from ..input_engineering.input_utils import (
    get_left_and_right_pos,
    check_pred_len,
    get_ed_candidates_per_item,
    get_eae_candidates,
    get_event_preds,
    get_plain_label,
)
logger = logging.getLogger(__name__)


def get_ace2005_trigger_detection_sl(preds: List[str],
                                     labels: List[str],
                                     data_file: str,
                                     data_args,
                                     is_overflow) -> List[str]:
    """Obtains the event detection prediction results of the ACE2005 dataset based on the sequence labeling paradigm.

    Obtains the event detection prediction results of the ACE2005 dataset based on the sequence labeling paradigm,
    predicting the labels and calculating the micro F1 score based on the predictions and labels.

    Args:
        preds (`List[str]`):
            A list of strings indicating the predicted types of the instances.
        labels (`List[str]`):
            A list of strings indicating the actual labels of the instances.
        data_file (`str`):
            A string indicating the path of the testing data file.
        data_args:
            The pre-defined arguments for data processing.
        is_overflow:


    Returns:
        results (`List[str]`):
            A list of strings indicating the prediction results of event triggers.
    """
    # get per-word predictions
    preds, labels = select_start_position(preds, labels, False)
    results = []
    label_names = []
    language = data_args.language

    with open(data_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())

            if not is_overflow[i]:
                check_pred_len(pred=preds[i], item=item, language=language)

            candidates, label_names_per_item = get_ed_candidates_per_item(item=item)
            label_names.extend(label_names_per_item)

            # loop for converting
            for candidate in candidates:
                left_pos, right_pos = get_left_and_right_pos(text=item["text"], trigger=candidate, language=language)
                pred = get_pred_per_mention(left_pos, right_pos, preds[i], data_args.id2type)
                results.append(pred)

    if "events" in item:
        micro_f1 = compute_unified_micro_f1(label_names=label_names, results=results)
        logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))

    return results


def get_ace2005_argument_extraction_sl(preds: List[str],
                                       labels: List[str],
                                       data_file: str,
                                       data_args,
                                       is_overflow) -> List[str]:
    """Obtains the event argument extraction results of the ACE2005 dataset based on the sequence labeling paradigm.

    Obtains the event argument extraction prediction results of the ACE2005 dataset based on the sequence labeling
    paradigm, predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
    predictions and labels.

    Args:
        preds (`List[str]`):
            A list of strings indicating the predicted types of the instances.
        labels (`List[str]`):
            A list of strings indicating the actual labels of the instances.
        data_file (`str`):
            A string indicating the path of the testing data file.
        data_args:
            The pre-defined arguments for data processing.
        is_overflow:


    Returns:
        results (`List[str]`):
            A list of strings indicating the prediction results of event arguments.
    """
    # evaluation mode
    eval_mode = data_args.eae_eval_mode
    language = data_args.language
    golden_trigger = data_args.golden_trigger

    # pred events
    event_preds = get_event_preds(pred_file=data_args.test_pred_file)

    # get per-word predictions
    preds, labels = select_start_position(preds, labels, False)
    results = []
    label_names = []
    with open(data_file, "r", encoding="utf-8") as f:
        trigger_idx = 0
        eae_instance_idx = 0
        lines = f.readlines()
        for line in lines:
            item = json.loads(line.strip())
            text = item["text"]
            for event in item["events"]:
                for trigger in event["triggers"]:
                    true_type = event["type"]
                    pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]

                    trigger_idx += 1

                    if eval_mode in ['default', 'loose']:
                        if pred_type == "NA":
                            continue

                    if not is_overflow[eae_instance_idx]:
                        check_pred_len(pred=preds[eae_instance_idx], item=item, language=language)

                    candidates, label_names_per_trigger = get_eae_candidates(item=item, trigger=trigger)
                    label_names.extend(label_names_per_trigger)

                    # loop for converting
                    for candidate in candidates:
                        # get word positions
                        left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candidate, language=language)
                        # get predictions
                        pred = get_pred_per_mention(left_pos, right_pos, preds[eae_instance_idx], data_args.id2role)
                        # record results
                        results.append(pred)
                    eae_instance_idx += 1

            # negative triggers
            for trigger in item["negative_triggers"]:
                true_type = "NA"
                pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                trigger_idx += 1

                if eval_mode in ['default', 'strict']:  # loose mode has no neg
                    if pred_type != "NA":
                        if not is_overflow[eae_instance_idx]:
                            check_pred_len(pred=preds[eae_instance_idx], item=item)

                        candidates = []
                        for neg in item["negative_triggers"]:
                            label_names.append("NA")
                            candidates.append(neg)

                        # loop for converting
                        for candidate in candidates:
                            # get word positions
                            left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candidate, language=language)
                            # get predictions
                            pred = get_pred_per_mention(left_pos, right_pos, preds[eae_instance_idx], data_args.id2role)
                            # record results
                            results.append(pred)

                        eae_instance_idx += 1

        assert len(preds) == eae_instance_idx
        
    pos_labels = list(set(label_names))
    pos_labels.remove("NA")
    micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0

    logger.info('Number of Instances: {}'.format(eae_instance_idx))
    logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))
    return results
<<<<<<< HEAD
=======


def get_ace2005_argument_extraction_mrc(preds, labels, data_file, data_args, is_overflow):
    # evaluation mode
    eval_mode = data_args.eae_eval_mode
    golden_trigger = data_args.golden_trigger
    language = data_args.language

    # pred events
    event_preds = get_event_preds(pred_file=data_args.test_pred_file)

    # get per-word predictions
    results = []
    all_labels = []
    with open(data_args.test_file, "r", encoding="utf-8") as f:
        trigger_idx = 0
        eae_instance_idx = 0
        lines = f.readlines()
        for line in lines:
            item = json.loads(line.strip())
            text = item["text"]

            # preds per index 
            preds_per_idx = []
            for pred in preds:
                if pred[-1] == trigger_idx:
                    preds_per_idx.append(pred)

            for event in item["events"]:
                for trigger in event["triggers"]:
                    true_type = event["type"]
                    pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                    trigger_idx += 1

                    if eval_mode in ['default', 'loose']:
                        if pred_type == "NA":
                            continue

                    # get candidates 
                    candidates, labels_per_idx = get_eae_candidates(item, trigger)
                    all_labels.extend(labels_per_idx)

                    # loop for converting
                    for candidate in candidates:
                        # get word positions
                        left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candidate, language=language)
                        # get predictions
                        pred_type = "NA"
                        for pred in preds_per_idx:
                            if pred[1] == (left_pos, right_pos):
                                pred_type = pred[0].split("_")[0]
                        # record results
                        results.append(pred_type)
                    eae_instance_idx += 1

            # negative triggers
            for trigger in item["negative_triggers"]:
                true_type = "NA"
                pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                trigger_idx += 1

                if eval_mode in ['default', 'strict']:  # loose mode has no neg
                    if pred_type != "NA":
                        candidates = []
                        for neg in item["negative_triggers"]:
                            candidates.append(neg)

                        # loop for converting
                        for candidate in candidates:
                            # get word positions
                            left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candidate, language=language)

                            # get predictions
                            pred_type = "NA"
                            for pred in preds_per_idx:
                                if pred[1] == (left_pos, right_pos):
                                    pred_type = pred[0].split("_")[0]
                            # record results
                            results.append(pred_type)

                        eae_instance_idx += 1

        assert len(preds) == eae_instance_idx
        
    pos_labels = list(data_args.role2id.keys())
    pos_labels.remove("NA")
    micro_f1 = f1_score(all_labels, results, labels=pos_labels, average="micro") * 100.0

    logger.info('Number of Instances: {}'.format(eae_instance_idx))
    logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))
    return results


def get_ace2005_trigger_detection_s2s(preds, labels, data_file, data_args, is_overflow):
    # get per-word predictions
    results = []
    label_names = []
    with open(data_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            preds_per_idx = preds[idx]

            candidates, labels_per_item = get_ed_candidates_per_item(item=item)
            for i, label in enumerate(labels_per_item):
                labels_per_item[i] = get_plain_label(label)
            label_names.extend(labels_per_item)

            # loop for converting 
            for candidate in candidates:
                pred_type = "NA"

                word = candidate["trigger_word"]
                if word in preds_per_idx and preds_per_idx[word] in data_args.type2id:
                    pred_type = preds_per_idx[word]

                results.append(pred_type)

    if "events" in item:
        micro_f1 = compute_unified_micro_f1(label_names=label_names, results=results)
        logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))

    return results


def get_ace2005_argument_extraction_s2s(preds, labels, data_file, data_args, is_overflow):
    # evaluation mode
    eval_mode = data_args.eae_eval_mode
    golden_trigger = data_args.golden_trigger

    # pred events
    event_preds = get_event_preds(pred_file=data_args.test_pred_file)

    # get per-word predictions
    results = []
    all_labels = []
    with open(data_args.test_file, "r", encoding="utf-8") as f:
        trigger_idx = 0
        eae_instance_idx = 0
        lines = f.readlines()
        for line in lines:
            item = json.loads(line.strip())
            # preds per index 
            preds_per_idx = []
            for pred in preds:
                if pred[-1] == trigger_idx:
                    preds_per_idx.append(pred)

            for event in item["events"]:
                for trigger in event["triggers"]:
                    true_type = event["type"]
                    pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                    trigger_idx += 1

                    if eval_mode in ['default', 'loose']:
                        if pred_type == "NA":
                            continue

                    # get candidates 
                    candidates, labels_per_idx = get_eae_candidates(item, trigger)
                    all_labels.extend(labels_per_idx)

                    # loop for converting
                    for candidate in candidates:
                        # get word positions
                        char_pos = candidate["position"]

                        # get predictions
                        candidate_mention = item["text"][char_pos[0]:char_pos[1]]
                        pred_type = "NA"
                        for pred in preds_per_idx:
                            if pred[-1] == candidate_mention:
                                pred_type = pred[-2]
                        # record results
                        results.append(pred_type)
                    eae_instance_idx += 1

            # negative triggers
            for trigger in item["negative_triggers"]:
                true_type = "NA"
                pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                trigger_idx += 1

                if eval_mode in ['default', 'strict']:  # loose mode has no neg
                    if pred_type != "NA":
                        candidates = []
                        for neg in item["negative_triggers"]:
                            candidates.append(neg)

                        # loop for converting
                        for candidate in candidates:
                            # get word positions
                            char_pos = candidate["position"]

                            # get predictions
                            candidate_mention = item["text"][char_pos[0]:char_pos[1]]
                            pred_type = "NA"
                            for pred in preds_per_idx:
                                if pred[-1] == candidate_mention:
                                    pred_type = pred[-2]
                            # record results
                            results.append(pred_type)

                        eae_instance_idx += 1

        assert len(preds) == eae_instance_idx
        
    pos_labels = list(data_args.role2id.keys())
    pos_labels.remove("NA")
    micro_f1 = f1_score(all_labels, results, labels=pos_labels, average="micro") * 100.0

    logger.info("Number of Instances: {}".format(eval_mode, eae_instance_idx))
    logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))
    return results
>>>>>>> dev
