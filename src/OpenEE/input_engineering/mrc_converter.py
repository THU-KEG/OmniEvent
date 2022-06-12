import os 
import pdb 
import json

import collections

def read_query_templates(prompt_file):
    """Load query templates"""
    query_templates = dict()
    with open(prompt_file, "r", encoding='utf-8') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")

            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if arg_name not in query_templates[event_type]:
                query_templates[event_type][arg_name] = list()

            # 0 template arg_name
            query_templates[event_type][arg_name].append(arg_name)
            # 1 template arg_name + in trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(arg_name + " in [trigger]")
            # 2 template arg_query
            query_templates[event_type][arg_name].append(query)
            # 3 arg_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    # with open(des_file, "r", encoding='utf-8') as f:
    #     for line in f:
    #         event_arg, query = line.strip().split(",")
    #         event_type, arg_name = event_arg.split("_")
    #         # 4 template des_query
    #         query_templates[event_type][arg_name].append(query)
    #         # 5 template des_query + trigger (replace [trigger] when forming the instance)
    #         query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    for event_type in query_templates:
        for arg_name in query_templates[event_type]:
            assert len(query_templates[event_type][arg_name]) == 4

    return query_templates


def _get_best_indexes(logits, n_best_size=1, larger_than_cls=False, cls_logit=None):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        if larger_than_cls:
            if index_and_score[i][1] < cls_logit:
                break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def char_pos_to_word_pos(text, position):
    return len(text[:position].split())


def make_preditions(all_start_logits, all_end_logits, training_args):
    data_for_evaluation = training_args.data_for_evaluation
    assert len(data_for_evaluation["golden_arguments"]) == len(all_start_logits)
    _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                               ["start_index", "end_index", "start_logit", "end_logit"])
    final_all_predictions = []
    for i, (start_logits, end_logits) in enumerate(zip(all_start_logits, all_end_logits)):
        event_argument_type = data_for_evaluation["golden_arguments"][i]["pred_type"] + "_" + data_for_evaluation["golden_arguments"][i]["role"]
        start_indexes = _get_best_indexes(start_logits, 20, False, start_logits[0])
        end_indexes = _get_best_indexes(end_logits, 20, False, end_logits[0])
        # add span preds
        prelim_predictions = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index == end_index and start_index == 0:
                    prelim_predictions.append(
                    _PrelimPrediction(start_index=-1, end_index=-1,
                                        start_logit=start_logits[start_index], end_logit=end_logits[end_index]))
                    continue 
                if start_index >= data_for_evaluation["text_range"][i]["end"] or \
                    end_index >= data_for_evaluation["text_range"][i]["end"]:
                    continue
                if start_index not in data_for_evaluation["subword_to_word"][i] or \
                    end_index not in data_for_evaluation["subword_to_word"][i]:
                    continue
                if end_index < start_index:
                    continue


                word_start_index = data_for_evaluation["subword_to_word"][i][start_index]
                word_end_index = data_for_evaluation["subword_to_word"][i][end_index]
                length = word_end_index - word_start_index + 1
                if length > 6:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(start_index=word_start_index, end_index=word_end_index+1,
                                        start_logit=start_logits[start_index], end_logit=end_logits[end_index]))
        ## sort
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        ## get final pred in format: [event_type_offset_argument_type, [start_offset, end_offset]]
        max_num_pred_per_arg = 4
        predictions_per_query = []
        for idx, pred in enumerate(prelim_predictions):
            if (idx + 1) > max_num_pred_per_arg: 
                break
            if pred.start_index == -1 and pred.end_index == -1: 
                break
            na_prob = (start_logits[0] + end_logits[0]) - (pred.start_logit + pred.end_logit)
            predictions_per_query.append([event_argument_type, [pred.start_index, pred.end_index], na_prob])
        final_all_predictions.extend(predictions_per_query)

    # all golden labels
    final_all_labels = []
    for i, arguments in enumerate(data_for_evaluation["golden_arguments"]):
        event_argument_type = data_for_evaluation["golden_arguments"][i]["true_type"] + "_" + arguments["role"]
        arguments_per_trigger = []
        for argument in arguments["arguments"]:
            for mention in argument["mentions"]:
                word_start_pos = char_pos_to_word_pos(data_for_evaluation["text"][i], mention["position"][0])
                word_length = len(mention["mention"].split())
                arguments_per_trigger.append([event_argument_type, [word_start_pos, word_start_pos+word_length]])
        final_all_labels.extend(arguments_per_trigger)
    
    return final_all_predictions, final_all_labels
        

def find_best_thresh(new_preds, new_all_gold):
    best_score = 0
    best_na_thresh = 0
    gold_arg_n, pred_arg_n = len(new_all_gold), 0

    candidate_preds = []
    for argument in new_preds:
        candidate_preds.append(argument[:-1])
        pred_arg_n += 1

        pred_in_gold_n, gold_in_pred_n = 0, 0
        # pred_in_gold_n
        for argu in candidate_preds:
            if argu in new_all_gold:
                pred_in_gold_n += 1
        # gold_in_pred_n
        for argu in new_all_gold:
            if argu in candidate_preds:
                gold_in_pred_n += 1

        prec_c, recall_c, f1_c = 0, 0, 0
        if pred_arg_n != 0: prec_c = 100.0 * pred_in_gold_n / pred_arg_n
        else: prec_c = 0
        if gold_arg_n != 0: recall_c = 100.0 * gold_in_pred_n / gold_arg_n
        else: recall_c = 0
        if prec_c or recall_c: f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
        else: f1_c = 0

        if f1_c > best_score:
            best_score = f1_c
            best_na_thresh = argument[-1]

    # import ipdb; ipdb.set_trace()
    return best_na_thresh + 1e-10


def compute_mrc_F1_cls(all_predcitions, all_labels):
    all_predcitions = sorted(all_predcitions, key=lambda x: x[-1])
    # best_na_thresh = 0
    best_na_thresh = find_best_thresh(all_predcitions, all_labels)

    final_new_preds = []
    for argument in all_predcitions:
        if argument[-1] < best_na_thresh:
            final_new_preds.append(argument[:-1]) # no na_prob

    # pdb.set_trace()
    # get results (classification)
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    # pred_arg_n
    for argument in final_new_preds: 
        pred_arg_n += 1
    # gold_arg_n     
    for argument in all_labels: 
        gold_arg_n += 1
    # pred_in_gold_n
    for argument in final_new_preds:
        if argument in all_labels:
            pred_in_gold_n += 1
    # gold_in_pred_n
    for argument in all_labels:
        if argument in final_new_preds:
            gold_in_pred_n += 1

    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_arg_n != 0:
         prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else: 
        prec_c = 0
    if gold_arg_n != 0: 
        recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else: 
        recall_c = 0
    if prec_c or recall_c: 
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else: f1_c = 0

    return f1_c









    