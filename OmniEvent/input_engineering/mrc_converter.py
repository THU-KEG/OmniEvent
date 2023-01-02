import collections
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_query_templates(prompt_file: str,
                         translate: Optional[bool] = False) -> Dict[str, Dict[str, List[str]]]:
    """Loads query templates from a prompt file.

    Loads query templates from a prompt file. If a translation is required, the query templates would be translated from
    English to Chinese based on four types of regulations.

    Args:
        prompt_file (`str`):
            A string indicating the path of the prompt file.
        translate (`bool`, `optional`, defaults to `False`):
            A boolean variable indicating whether or not to translate the query templates into Chinese.

    Returns:
        query_templates (`Dict[str, Dict[str, List[str]]]`)
            A dictionary containing the query templates applicable for every event type and argument role.
    """
    et_translation = dict()
    ar_translation = dict()
    if translate:
        # the event types and argument roles in ACE2005-zh are expressed in English, we translate them to Chinese
        et_file = "/".join(prompt_file.split('/')[:-1]) + "/chinese_event_types.txt"
        title = None

        for line in open(et_file, encoding='utf-8').readlines():
            num = line.split()[0]
            chinese = line.split()[1][:line.split()[1].index("（")]
            english = line[line.index("（") + 1:line.index('）')]
            if '.' not in num:
                title = chinese, english
            if title:
                et_translation['{}.{}'.format(title[1], english)] = "{}.{}".format(title[0], chinese)

        ar_file = "/".join(prompt_file.split('/')[:-1]) + "/chinese_arg_roles.txt"
        for line in open(ar_file, encoding='utf-8').readlines():
            english, chinese = line.strip().split()
            ar_translation[english] = chinese

    query_templates = dict()
    with open(prompt_file, "r", encoding='utf-8') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")

            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if arg_name not in query_templates[event_type]:
                query_templates[event_type][arg_name] = list()

            if translate:
                # 0 template arg_name
                query_templates[event_type][arg_name].append(ar_translation[arg_name])
                # 1 template arg_name + in trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(ar_translation[arg_name] + "在[trigger]中")
                # 2 template arg_query
                query_templates[event_type][arg_name].append(query)
                # 3 arg_query + trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(query[:-1] + "在[trigger]中?")
            else:
                # 0 template arg_name
                query_templates[event_type][arg_name].append(arg_name)
                # 1 template arg_name + in trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(arg_name + " in [trigger]")
                # 2 template arg_query
                query_templates[event_type][arg_name].append(query)
                # 3 arg_query + trigger (replace [trigger] when forming the instance)
                query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    for event_type in query_templates:
        for arg_name in query_templates[event_type]:
            assert len(query_templates[event_type][arg_name]) == 4

    return query_templates


def _get_best_indexes(logits: List[int],
                      n_best_size: Optional[int] = 1,
                      larger_than_cls: Optional[bool] = False,
                      cls_logit: Optional[int] = None) -> List[int]:
    """Gets the n-best logits from a list.

    Gets the n-best logits from a list. The methods returns a list containing the indexes of the n-best logits that
    satisfies both the logits are n-best and greater than the logit of the "cls" token.
    """
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


def char_pos_to_word_pos(text: str,
                         position: int) -> int:
    """Returns the word-level position of a mention.

    Returns the word-level position of a mention by counting the number of words before the start position of the
    mention.

    Args:
        text (`str`):
            A string representing the source text that the mention is within.
        position (`int`)
            An integer indicating the character-level position of the mention.

    Returns:
        An integer indicating the word-level position of the mention.
    """
    return len(text[:position].split())


def make_predictions(all_start_logits, all_end_logits, training_args, use_example_id=True):
    """Obtains the prediction from the Machine Reading Comprehension (MRC) model."""
    data_for_evaluation = training_args.data_for_evaluation
    assert len(all_start_logits) == len(data_for_evaluation["ids"])
    # all golden labels
    final_all_labels = []
    for arguments in data_for_evaluation["golden_arguments"]:
        arguments_per_trigger = []
        for argument in arguments["arguments"]:
            event_argument_type = arguments["true_type"] + "_" + argument["role"]
            for mention in argument["mentions"]:
                arguments_per_trigger.append(
                    (event_argument_type, (mention["position"][0], mention["position"][1]), arguments["id"]))
        final_all_labels.extend(arguments_per_trigger)
    # predictions
    _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                               ["start_index", "end_index", "start_logit", "end_logit"])
    final_all_predictions = []
    for example_id, (start_logits, end_logits) in enumerate(zip(all_start_logits, all_end_logits)):
        event_argument_type = data_for_evaluation["pred_types"][example_id] + "_" + \
                              data_for_evaluation["roles"][example_id]
        start_indexes = _get_best_indexes(start_logits, 20, True, start_logits[0])
        end_indexes = _get_best_indexes(end_logits, 20, True, end_logits[0])
        # add span preds
        prelim_predictions = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index < data_for_evaluation["text_range"][example_id]["start"] or \
                        end_index < data_for_evaluation["text_range"][example_id]["start"]:
                    continue
                if start_index >= data_for_evaluation["text_range"][example_id]["end"] or \
                        end_index >= data_for_evaluation["text_range"][example_id]["end"]:
                    continue
                if end_index < start_index:
                    continue
                word_start_index = start_index - 1
                word_end_index = end_index - 1
                length = word_end_index - word_start_index + 1
                if length > 5:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(start_index=word_start_index, end_index=word_end_index,
                                      start_logit=start_logits[start_index], end_logit=end_logits[end_index]))
        # sort
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
        # get final pred in format: [event_type_offset_argument_type, [start_offset, end_offset]]
        max_num_pred_per_arg = 1
        predictions_per_query = []
        for _, pred in enumerate(prelim_predictions[:max_num_pred_per_arg]):
            na_prob = (start_logits[0] + end_logits[0]) - (pred.start_logit + pred.end_logit)
            predictions_per_query.append((event_argument_type, (pred.start_index, pred.end_index), na_prob,
                                          data_for_evaluation["ids"][example_id] if use_example_id else data_for_evaluation["trigger_ids"][example_id]))
        final_all_predictions.extend(predictions_per_query)

    logger.info("\nAll predictions and labels generated. %d %d\n" % (len(final_all_predictions), len(final_all_labels)))
    return final_all_predictions, final_all_labels


def find_best_thresh(new_preds, new_all_gold):
    best_score = 0
    best_na_thresh = 0
    gold_arg_n, pred_arg_n = len(new_all_gold), 0

    candidate_preds = []
    for argument in tqdm(new_preds, desc="Finding best thresh"):
        candidate_preds.append(argument[:-2] + argument[-1:])
        pred_arg_n += 1

        pred_in_gold_n = len(set(candidate_preds).intersection(set(new_all_gold)))
        gold_in_pred_n = len(set(new_all_gold).intersection(set(candidate_preds)))

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
        else:
            f1_c = 0

        if f1_c > best_score:
            best_score = f1_c
            best_na_thresh = argument[-2]

    return best_na_thresh + 1e-10


def compute_mrc_F1_cls(all_predictions, all_labels):
    all_predictions = sorted(all_predictions, key=lambda x: x[-2])
    # best_na_thresh = 0
    best_na_thresh = find_best_thresh(all_predictions, all_labels)
    print("Best thresh founded. %.6f" % best_na_thresh)

    final_new_preds = []
    for argument in all_predictions:
        if argument[-2] < best_na_thresh:
            final_new_preds.append(argument[:-2] + argument[-1:])  # no na_prob

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
    else:
        f1_c = 0

    # logger.info("Precision: %.2f, recall: %.2f" % (prec_c, recall_c))
    return prec_c, recall_c, f1_c
