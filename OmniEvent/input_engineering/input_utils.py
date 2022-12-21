import os
import json
import logging

from pathlib import Path
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding
from .whitespace_tokenizer import WordLevelTokenizer
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


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
                           language: str,
                           keep_space: bool = False) -> Tuple[int, int]:
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
        keep_space (`bool`):
            A flag that indicates whether to keep the space in Chinese text during offset calculating.
                During data preprocessing, the space has to be kept due to the offsets consider space.
                During evaluation, the space is automatically removed by the tokenizer and the output hidden states do
                not involve space logits, therefore, offset counting should not keep the space.
    Returns:
        left_pos (`int`), right_pos (`int`):
            Two integers indicating the number of words before the start and end position of the trigger word.
    """
    if language == "English":
        left_pos = len(text[:trigger["position"][0]].split())
        right_pos = len(text[:trigger["position"][1]].split())
    elif language == "Chinese":
        left_pos = trigger["position"][0] if keep_space else len("".join(text[:trigger["position"][0]].split()))
        right_pos = trigger["position"][1] if keep_space else len("".join(text[:trigger["position"][1]].split()))
    else:
        raise NotImplementedError
    return left_pos, right_pos


def char_pos_to_word_pos(text: str,
                           char_pos: List[int],
                           language: str = "English",
                           keep_space: bool = False) -> Tuple[int, int]:
    if language == "English":
        left_pos = len(text[:char_pos[0]].split())
        right_pos = len(text[:char_pos[1]].split())
    elif language == "Chinese":
        left_pos = char_pos[0] if keep_space else len("".join(text[:char_pos[0]].split()))
        right_pos = char_pos[1] if keep_space else len("".join(text[:char_pos[1]].split()))
    else:
        raise NotImplementedError
    return left_pos, right_pos


def get_word_ids(tokenizer: PreTrainedTokenizer,
                 outputs: BatchEncoding,
                 word_list: List[str]) -> List[int]:
    """Return a list mapping the tokens to their actual word in the initial sentence for a tokenizer.

    Return a list indicating the word corresponding to each token. Special tokens added by the tokenizer are mapped to
    None and other tokens are mapped to the index of their corresponding word (several tokens will be mapped to the same
    word index if they are parts of that word).

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer that has been used for word tokenization.
        outputs (`BatchEncoding`):
            The outputs of the tokenizer.
        word_list (`List[str]`):
            A list of word strings.
    Returns:
        word_ids (`List[int]`):
            A list mapping the tokens to their actual word in the initial sentence
    """
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

    for token in tokens:
        if token not in word_list and token != "[UNK]":
            word_ids.append(None)
        else:
            if token != "[UNK]":
                assert token == word_list[word_idx]
            word_ids.append(word_idx)
            word_idx += 1
    return word_ids


def check_pred_len(pred: List[str],
                   item: Dict[str, Union[str, List[dict]]],
                   language: str) -> None:
    """Check whether the length of the prediction sequence equals that of the original word sequence.

    The prediction sequence consists of prediction for each word in the original sentence. Sometimes, there might be
    special tokens or extra space in the original sentence, and the tokenizer will automatically ignore them, which may
    cause the output length differs from the input length.

    Args:
        pred (`List[str]`):
            A list of predicted event types or argument roles.
        item (`Dict[str, Union[str, List[dict]]]`):
            A single item of the training/valid/test data.
        language ('str'):
            The language of the input text.
    Returns:
        None.
    """
    if language == "English":
        if len(pred) != len(item["text"].split()):
            logger.warning("There might be special tokens in the input text: {}".format(item["text"]))

    elif language == "Chinese":
        if len(pred) != len("".join(item["text"].split())):  # remove space token
            logger.warning("There might be special tokens in the input text: {}".format(item["text"]))
    else:
        raise NotImplementedError


def get_ed_candidates(item: Dict[str, Union[str, List[dict]]]) -> Tuple[List[dict], List[str]]:
    """Obtain the candidate tokens for the event detection (ED) task.

    The unified evaluation considers prediction of each token that is possibly a trigger (ED candidate).

    Args:
        item (`Dict[str, Union[str, List[dict]]]`):
            A single item of the training/valid/test data.
    Returns:
        candidates(`List[dict]`), label_names (`List[str]`):
            candidates: A list of dictionary that contains the possible trigger.
            label_names: A list of string contains the ground truth label for each possible trigger.
    """
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
        label_names = ["NA"] * len(candidates)

    return candidates, label_names


def check_is_argument(mention: Dict[str, Union[str, dict]] = None,
                      positive_offsets: List[Tuple[int, int]] = None) -> bool:
    """Check whether a given mention is argument or not.

    Check whether a given mention is argument or not. If it is an argument, we have to exclude it from the negative
    arguments list.

    Args:
        mention (`Dict[str, Union[str, dict]]`):
            The mention that contains the word, position and other meta information like id, etc.
        positive_offsets (`List[Tuple[int, int]]`):
            A list that contains the offsets of all the ground truth arguments.
    Returns:
        is_argument(`bool`):
            A flag that indicates whether the mention is an argument or not.

    """
    is_argument = False
    if positive_offsets:
        mention_set = set(range(mention["position"][0], mention["position"][1]))
        for pos_offset in positive_offsets:
            pos_set = set(range(pos_offset[0], pos_offset[1]))
            if not pos_set.isdisjoint(mention_set):
                is_argument = True
                break
    return is_argument


def get_negative_argument_candidates(item: Dict[str, Union[str, List[dict]]],
                                     positive_offsets: List[Tuple[int, int]] = None,
                                     ) -> List[Dict[str, Union[str, dict]]]:
    """Obtain the negative candidate arguments for each trigger in the event argument extraction (EAE) task.

    Obtain the negative candidate arguments, which are not included in the actual arguments list, for the specified
    trigger. The unified evaluation considers prediction of each token that is possibly an argument (EAE candidate).

    Args:
        item (`Dict[str, Union[str, List[dict]]]`):
            A single item of the training/valid/test data.
        positive_offsets (`List[Tuple[int, int]]`):
            A list that contains the offsets of all the ground truth arguments.
    Returns:
        candidates(`List[dict]`), label_names (`List[str]`):
            candidates: A list of dictionary that contains the possible arguments.
            label_names: A list of string contains the ground truth label for each possible argument.
    """
    if "entities" in item:
        neg_arg_candidates = []
        for entity in item["entities"]:
            ent_is_arg = any([check_is_argument(men, positive_offsets) for men in entity["mentions"]])
            neg_arg_candidates.extend([] if ent_is_arg else entity["mentions"])
    else:
        neg_arg_candidates = item["negative_triggers"]
    return neg_arg_candidates


def get_eae_candidates(item: Dict[str, Union[str, List[dict]]],
                       trigger: Dict[str, Union[str, dict]]) -> Tuple[List[dict], List[str]]:
    """Obtain the candidate arguments for each trigger in the event argument extraction (EAE) task.

    The unified evaluation considers prediction of each token that is possibly an argument (EAE candidate). And the EAE
    task requires the model to predict the argument role of each candidate given a specific trigger.

    Args:
        item (`Dict[str, Union[str, List[dict]]]`):
            A single item of the training/valid/test data.
        trigger (`Dict[str, Union[str, List[dict]]`):
            A single item of trigger in the item.
    Returns:
        candidates(`List[dict]`), label_names (`List[str]`):
            candidates: A list of dictionary that contains the possible arguments.
            label_names: A list of string contains the ground truth label for each possible argument.
    """
    candidates = []
    positive_offsets = []
    label_names = []
    if "arguments" in trigger:
        arguments = sorted(trigger["arguments"], key=lambda a: a["role"])
        for argument in arguments:
            for mention in argument["mentions"]:
                label_names.append(argument["role"])
                candidates.append(mention)
                positive_offsets.append(mention["position"])

    neg_arg_candidates = get_negative_argument_candidates(item, positive_offsets=positive_offsets)

    for neg in neg_arg_candidates:
        is_argument = check_is_argument(neg, positive_offsets)
        if not is_argument:
            label_names.append("NA")
            candidates.append(neg)

    return candidates, label_names


def get_event_preds(pred_file: Union[str, Path]) -> List[str]:
    """Load the event detection predictions of each token for event argument extraction.

    The Event Argument Extraction task requires the event detection predictions. If the event prediction file exists,
    we use the predictions by the event detection model. Otherwise, we use the golden event type for each token.

    Args:
        pred_file (`Union[str, Path]`):
            The file that contains the event detection predictions for each token.
    Returns:
        event_preds (`List[str`]):
            A list of the predicted event types for each token.
    """
    if pred_file is not None and os.path.exists(pred_file):
        event_preds = json.load(open(pred_file))
    else:
        event_preds = None
        logger.info("Load {} failed, using golden triggers for EAE evaluation".format(pred_file))

    return event_preds


def get_plain_label(input_label: str) -> str:
    """Convert the formatted original event type or argument role to a plain one.

    This function is used in the Seq2seq paradigm that the model has to generate the event types and argument roles.
    Some  event types and argument roles are formatted, such as `Attack.Time-Start`, we convert them in to a plain
    one, like `timestart`, by removing the event type in the front and shifting upper case to lower case.

    Args:
        input_label (`str`):
            The original label with format.
    Returns:
        return_label (`str`):
            The plain label without format.

    """
    if input_label == "NA":
        return input_label

    return_label = "".join("".join(input_label.split(".")[-1].split("-")).split("_")).lower()

    return return_label


def str_full_to_half(ustring: str) -> str:
    """Convert a full-width string to a half-width one.

    The corpus of some datasets contain full-width strings, which may bring about unexpected error for mapping the
    tokens to the original input sentence.

    Args:
        ustring(`str`):
            Original string.
    Returns:
        rstring (`str`):
            Output string with the full-width tokens converted
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:   # full width space
            inside_code = 32
        elif 65281 <= inside_code <= 65374:    # full width char (exclude space)
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
