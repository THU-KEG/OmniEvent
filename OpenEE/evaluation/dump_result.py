import os
import sys
import pdb 
import argparse
from typing import List

import jsonlines
import json
import numpy as np
from tqdm import tqdm
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


def get_sentence_arguments(input_sentence):
    input_sentence.append({"role": "NA", "word": "<EOS>"})
    arguments = []

    previous_role = None
    previous_arg = ""
    for item in input_sentence:
        if item["role"] != "NA" and previous_role is None:
            previous_role = item["role"]
            previous_arg = item["word"]

        elif item["role"] == previous_role:
            previous_arg += item["word"]

        elif item["role"] != "NA":
            arguments.append({"role": previous_role, "argument": previous_arg})
            previous_role = item["role"]
            previous_arg = item["word"]

        elif previous_role is not None:
            arguments.append({"role": previous_role, "argument": previous_arg})
            previous_role = None
            previous_arg = ""

    return arguments


def get_maven_submission(preds: List[str],
                         instance_ids: List[str],
                         result_file: str) -> None:
    """Converts the predictions to the submission format of the MAVEN dataset.

    Converts the predictions to the submission format of the MAVEN dataset and dumps a prediction json file.

    Args:
        preds (`List[str]`):
            A list of strings indicating the predicted type of each instance.
        instance_ids (`List[str]`):
            A list of strings containing the id of each instance to be predicted.
    """
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
                    assert len(preds[i]) == len("".join(item["text"].split()))  # remove space/special token
                else:
                    raise NotImplementedError

            for candidate in item["candidates"]:
                # get word positions
                char_pos = candidate["position"]

                if config.language == "English":
                    word_pos_start = len(item["text"][:char_pos[0]].split())
                    word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                elif config.language == "Chinese":
                    word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                    word_pos_end = len("".join(item["text"][:char_pos[1]].split()))
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


def get_maven_submission_seq2seq(preds,
                                 labels,
                                 save_path,
                                 type2id,
                                 tokenizer,
                                 training_args,
                                 data_args):
    """Converts the predictions obtained by the sequence-to-sequence model to the MAVEN dataset's submission format.

    Converts the predictions to the submission format of the MAVEN dataset and dumps a prediction json file.


    """
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


def get_leven_submission(preds: List[str],
                         instance_ids: List[str],
                         result_file: str) -> None:
    """Converts the predictions to the submission format of the MAVEN dataset.

    Converts the predictions to the submission format of the MAVEN dataset and dumps a prediction json file.

    Args:
        preds (`List[str]`):
            A list of strings indicating the predicted type of each instance.
        instance_ids (`List[str]`):
            A list of strings containing the id of each instance to be predicted.
    """
    return get_maven_submission(preds, instance_ids, result_file)


def get_leven_submission_sl(preds, labels, is_overflow, result_file, type2id, config):
    return get_maven_submission_sl(preds, labels, is_overflow, result_file, type2id, config)


def get_leven_submission_seq2seq(preds, labels, save_path, type2id, tokenizer, training_args, data_args):
    return get_maven_submission_seq2seq(preds, labels, save_path, type2id, tokenizer, training_args, data_args)


def get_duee_submission():
    pass


def get_duee_submission_sl(preds, labels, is_overflow, result_file, config):
    # trigger predictions
    ed_preds = json.load(open(config.test_pred_file))

    # get per-word predictions
    preds, labels = select_start_position(preds, labels, False)
    all_results = []

    with open(config.test_file, "r", encoding='utf-8') as f:
        trigger_idx = 0
        example_idx = 0
        lines = f.readlines()
        for line in tqdm(lines, desc='Generating DuEE1.0 Submission Files'):
            item = json.loads(line.strip())

            item_id = item["id"]
            event_list = []

            for tid, trigger in enumerate(item["candidates"]):
                pred_event_type = ed_preds[trigger_idx]
                if pred_event_type != "NA":
                    if not is_overflow[example_idx]:
                        if config.language == "English":
                            assert len(preds[example_idx]) == len(item["text"].split())
                        elif config.language == "Chinese":
                            # print('len preds: {}'.format(len(preds[example_idx])))
                            # print('len clean text: {}'.format(len("".join(item["text"].split()))))
                            # print('text:{}'.format(item['text']))
                            assert len(preds[example_idx]) == len("".join(item["text"].split()))  # remove space token
                        else:
                            raise NotImplementedError

                    pred_event = dict(event_type=pred_event_type, arguments=[])
                    sentence_result = []
                    for cid, candidate in enumerate(item["candidates"]):
                        if cid == tid:
                            continue
                        char_pos = candidate["position"]
                        if config.language == "English":
                            word_pos_start = len(item["text"][:char_pos[0]].split())
                            word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                        elif config.language == "Chinese":
                            word_pos_start = len([w for w in item["text"][:char_pos[0]] if w.strip('\n\xa0� ')])
                            word_pos_end = len([w for w in item["text"][:char_pos[1]] if w.strip('\n\xa0� ')])
                        else:
                            raise NotImplementedError
                        # get predictions
                        pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[example_idx], config.id2role)
                        sentence_result.append({"role": pred, "word": candidate["trigger_word"]})

                    pred_event["arguments"] = get_sentence_arguments(sentence_result)
                    if pred_event["arguments"]:
                        event_list.append(pred_event)

                    example_idx += 1

                trigger_idx += 1

            all_results.append({"id": item_id, "event_list": event_list})

    # dump results
    with jsonlines.open(result_file, "w") as f:
        for r in all_results:
            jsonlines.Writer.write(f, r)

    return all_results

