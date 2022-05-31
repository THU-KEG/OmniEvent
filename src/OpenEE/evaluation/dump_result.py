import os
import sys
import pdb 
import argparse
import json
import numpy as np
from collections import defaultdict
from .metric import select_start_position, compute_seq_F1
from ..input_engineering.input_utils import get_start_poses, check_if_start, get_word_position


def get_pred_per_mention(pos_start, pos_end, preds, id2label):
    if pos_start == pos_end or\
            pos_end > len(preds) or \
            id2label[int(preds[pos_start])] == "O" or \
            id2label[int(preds[pos_start])].split("-")[0] != "B":
        return "NA"
    predictions = set()
    for pos in range(pos_start, pos_end):
        _pred = id2label[int(preds[pos])][2:]
        predictions.add(_pred)
    if len(predictions) > 1:
        return "NA"
    return list(predictions)[0]


def get_maven_submission(preds, instance_ids, result_file):
    all_results = defaultdict(list)
    for i, pred in enumerate(preds):
        example_id, candidate_id = instance_ids[i].split("-")
        all_results[example_id].append({
            "id": candidate_id,
            "type_id": int(pred)
        })
    with open(result_file, "w") as f:
        for data_id in all_results.keys():
            format_result = dict(id=data_id, predictions=[])
            for candidate in all_results[data_id]:
                format_result["predictions"].append(candidate)
            f.write(json.dumps(format_result) + "\n")


def get_maven_submission_sl(preds, labels, is_overflow, result_file, type2id, config):
    # get per-word predictions
    preds, _ = select_start_position(preds, labels, False)
    results = defaultdict(list)
    with open(config.test_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())
            # check for alignment 
            if not is_overflow[i]:
                if config.language == "English":
                    assert len(preds[i]) == len(item["text"].split())
                elif config.language == "Chinese":
                    assert len(preds[i]) == len([w for w in item['text'] if w.strip()])  # remove space token
                else:
                    raise NotImplementedError

            for candidate in item["candidates"]:
                # get word positions
                char_pos = candidate["position"]

                if config.language == "English":
                    word_pos_start = len(item["text"][:char_pos[0]].split())
                    word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                elif config.language == "Chinese":
                    word_pos_start = len([w for w in item["text"][:char_pos[0]] if w.strip()])
                    word_pos_end = len([w for w in item["text"][:char_pos[1]] if w.strip()])
                else:
                    raise NotImplementedError

                # get predictions
                pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[i], config.id2type)
                # record results
                results[item["id"]].append({
                    "id": candidate["id"].split("-")[-1],
                    "type_id": int(type2id[pred]),
                })
    # dump results 
    with open(result_file, "w") as f:
        for id, preds_per_doc in results.items():
            results_per_doc = dict(id=id, predictions=preds_per_doc)
            f.write(json.dumps(results_per_doc)+"\n")


def get_maven_submission_seq2seq(preds, labels, save_path, type2id, tokenizer, training_args, data_args):
    decoded_preds = compute_seq_F1(preds, labels, 
                                    **{"tokenizer": tokenizer, 
                                       "training_args": training_args, 
                                       "return_decoded_preds": True})
    results = defaultdict(list)
    with open(data_args.test_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())
            for candidate in item["candidates"]:
                pred_type = "NA"
                # pdb.set_trace()
                if candidate["trigger_word"] in decoded_preds[i] and \
                    decoded_preds[i][candidate["trigger_word"]] in type2id:
                    pred_type = decoded_preds[i][candidate["trigger_word"]]
                # record results
                results[item["id"]].append({
                    "id": candidate["id"].split("-")[-1],
                    "type_id": int(type2id[pred_type]),
                })
    # dump results 
    with open(save_path, "w") as f:
        for id, preds_per_doc in results.items():
            results_per_doc = dict(id=id, predictions=preds_per_doc)
            f.write(json.dumps(results_per_doc)+"\n")


def get_leven_submission(preds, instance_ids, result_file):
    return get_maven_submission(preds, instance_ids, result_file)


def get_leven_submission_sl(preds, labels, is_overflow, result_file, type2id, config):
    return get_maven_submission_sl(preds, labels, is_overflow, result_file, type2id, config)


def get_leven_submission_seq2seq(preds, labels, save_path, type2id, tokenizer, training_args, data_args):
    return get_maven_submission_seq2seq(preds, labels, save_path, type2id, tokenizer, training_args, data_args)


def get_duee_ed_submission(preds, instance_ids, result_file, training_args):
    all_results = defaultdict(list)
    for i, pred in enumerate(preds):
        example_id, candidate_id = instance_ids[i].split("-")
        if pred != 0:
            all_results[example_id].append({"event_type": training_args.id2type[pred]})
    with open(result_file, "w", encoding='utf-8') as f:
        for data_id in all_results.keys():
            format_result = dict(id=data_id, event_list=[])
            for candidate in all_results[data_id]:
                format_result["event_list"].append(candidate)
            f.write(json.dumps(format_result, ensure_ascii=False) + "\n")


def get_duee_fin_ed_submission(preds, instance_ids, result_file, training_args):
    return get_duee_ed_submission(preds, instance_ids, result_file, training_args)
