import os
import json
import logging

from .whitespace_tokenizer import WordLevelTokenizer

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Union


def get_bio_labels(original_labels: List[str],
                   labels_to_exclude: Optional[List[str]] = ["NA"]) -> Dict[str, int]:
    """Generates the id of the BIO labels corresponding to the original label.

    Generates the id of the BIO labels corresponding to the original label. The correspondences between the BIO labels
    and their ids are saved in a dictionary.

    Args:
        original_labels (`List[str]`):
            A list of strings representing the original labels within the dataset.
        labels_to_exclude (`List[str]`, `optional`, defaults to ["NA"]):
            A list of strings indicating the labels excluded to use, the id of which would not be generated.

    Returns:
        bio_labels (`Dict[str, int]`):
            A dictionary containing the correspondence the BIO labels and their ids.
    """
    bio_labels = {"O": 0}
    for label in original_labels:
        if label in labels_to_exclude:
            continue
        bio_labels[f"B-{label}"] = len(bio_labels)
        bio_labels[f"I-{label}"] = len(bio_labels)
    return bio_labels


def get_start_poses(sentence: str) -> List[int]:
    """Obtains the start position of each word within the sentence.

    Obtains the start position of each word within the sentence. The character-level start positions of each word are
    stored in a list.

    Args:
        sentence (`str`):
            A string representing the input sentence.

    Returns:
        start_poses (`List[int]`):
            A list of integers representing the character-level start position of each word within the sentence.
    """
    words = sentence.split()
    start_pos = 0
    start_poses = []
    for word in words:
        start_poses.append(start_pos)
        start_pos += len(word) + 1
    return start_poses


def check_if_start(start_poses: List[int],
                   char_pos: List[int]) -> bool:
    """Check whether the start position of the mention is the beginning of a word.

    Check whether the start position of the mention is the beginning of a word, that is, check whether a trigger or an
    argument is a sub-word.

    Args:
        start_poses (`List[int]`):
            A list of integers representing the character-level start position of each word within the sentence.
        char_pos (`List[int]`):
            A list of integers indicating the start and end position of a mention.

    Returns:
        Returns `True` if the start position of the mention is the start of a word; returns `False` otherwise.
    """
    if char_pos[0] in start_poses:
        return True
    return False


def get_word_position(start_poses: List[int],
                      char_pos: List[int]) -> int:
    """Returns the word-level position of a given mention.

    Returns the word-level position of a given mention by matching the index of its character-level start position in
    the list containing the start position of each word within the sentence.

    Args:
        start_poses (`List[int]`):
            A list of integers representing the character-level start position of each word within the sentence.
        char_pos (`List[int]`)
            A list of integers indicating the start and end position of a given mention.

    Returns:
        `int`:
            An integer indicating the word-level position of the given mention.
    """
    return start_poses.index(char_pos[0])


def get_words(text: str,
              language: str) -> List[str]:
    """Obtains the words within the given text.

    Obtains the words within the source text. The recognition of words differs according to language. The words are
    obtained through splitting white spaces in English, while each Chinese character is regarded as a word in Chinese.

    Args:
        text (`str`):
            A string representing the input source text.
        language (`str`):
            A string indicating the language of the source text, English or Chinese.

    Returns:
        words (`List[str]`):
            A list of strings containing the words within the source text.
    """
    if language == "English":
        words = text.split()
    elif language == "Chinese":
        words = list(text)
    else:
        raise NotImplementedError
    return words


def get_left_and_right_pos(text: str,
                           trigger: Dict[str, Union[int, str, List[int], List[Dict]]],
                           language: str):
    """Obtains the word-level position of the trigger word's start and end position.

    Obtains the word-level position of the trigger word's start and end position. The method of obtaining the position
    differs according to language. The method returns the number of words before the given position for English texts,
    while for Chinese, each character is regarded as a word.

    Args:
        text (`str`):
            A string representing the source text that the trigger word is within.
        trigger (`Dict[str, Union[int, str, List[int], List[Dict]]]`):
            A dictionary containing the trigger word, position, and arguments of an event trigger.
        language (`str`):
            A string indicating the language of the source text and trigger word, English or Chinese.

    Returns:
        left_pos (`str`), right_pos (`str`):
            Two strings indicating the number of words before the start and end position of the trigger word.
    """
    if language == "English":
        left_pos = len(text[:trigger["position"][0]].split())
        right_pos = len(text[:trigger["position"][1]].split())
    elif language == "Chinese":
        left_pos = trigger["position"][0]
        right_pos = trigger["position"][1]
    else:
        raise NotImplementedError
    return left_pos, right_pos


def get_word_ids(tokenizer, outputs, word_list):
    word_list = [w.lower() for w in word_list]
    try:
        word_ids = outputs.word_ids()
        return word_ids
    except:
        assert isinstance(tokenizer, WordLevelTokenizer)
        pass
    tokens = tokenizer.convert_ids_to_tokens(outputs["input_ids"])
    word_ids = []
    word_idx = 0
    # import pdb; pdb.set_trace()
    for token in tokens:
        if token not in word_list and token != "[UNK]":
            word_ids.append(None)
        else:
            if token != "[UNK]":
                assert token == word_list[word_idx]
            word_ids.append(word_idx)
            word_idx += 1
    return word_ids


def check_pred_len(pred, item, language):
    if language == "English":
        if len(pred) != len(item["text"].split()):
            logger.warning("There might be special tokens in the input text: {}".format(item["text"]))

    elif language == "Chinese":
        if len(pred) != len("".join(item["text"].split())): # remove space token
            logger.warning("There might be special tokens in the input text: {}".format(item["text"]))
    else:
        raise NotImplementedError


def get_ed_candidates_per_item(item):
    candidates = []
    label_names = []
    if "events" in item:
        for event in item["events"]:
            for trigger in event["triggers"]:
                label_names.append(event["type"])
                candidates.append(trigger)
        for neg_trigger in item["negative_triggers"]:
            label_names.append("NA")
            candidates.append(neg_trigger)
    else:
        candidates = item["candidates"]

    return candidates, label_names


def get_eae_candidates(item, trigger):
    candidates = []
    positive_mentions = set()
    positive_offsets = []
    label_names = []
    for argument in trigger["arguments"]:
        for mention in argument["mentions"]:
            label_names.append(argument["role"])
            candidates.append(mention)
            positive_mentions.add(mention["mention_id"])
            positive_offsets.append(mention["position"])

    if "entities" in item:
        for entity in item["entities"]:
            # check whether the entity is an argument
            is_argument = False
            for mention in entity["mentions"]:
                if mention["mention_id"] in positive_mentions:
                    is_argument = True
                    break
            if is_argument:
                continue
            # negative arguments
            for mention in entity["mentions"]:
                label_names.append("NA")
                candidates.append(mention)
    else:
        for neg in item["negative_triggers"]:
            is_argument = False
            neg_set = set(range(neg["position"][0], neg["position"][1]))
            for pos_offset in positive_offsets:
                pos_set = set(range(pos_offset[0], pos_offset[1]))
                if not pos_set.isdisjoint(neg_set):
                    is_argument = True
                    break
            if is_argument:
                continue
            label_names.append("NA")
            candidates.append(neg)

    return candidates, label_names


def get_event_preds(pred_file):
    if pred_file is not None and os.path.exists(pred_file):
        event_preds = json.load(open(pred_file))
    else:
        event_preds = None
        logger.info("Load {} failed, using golden triggers for EAE evaluation".format(pred_file))

    return event_preds


def get_plain_label(input_label):
    if input_label == "NA":
        return input_label

    return "".join("".join(input_label.split(".")[-1].split("-")).split("_")).lower()
