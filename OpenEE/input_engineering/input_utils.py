from typing import Dict
from .whitespace_tokenizer import WordLevelTokenizer


def get_bio_labels(original_labels, labels_to_exclude=["NA"]) -> Dict[str, int]:
    bio_labels = {"O": 0}
    for label in original_labels:
        if label in labels_to_exclude:
            continue
        bio_labels[f"B-{label}"] = len(bio_labels)
        bio_labels[f"I-{label}"] = len(bio_labels)
    return bio_labels


def get_start_poses(sentence):
    words = sentence.split()
    start_pos = 0
    start_poses = []
    for word in words:
        start_poses.append(start_pos)
        start_pos += len(word) + 1
    return start_poses


def check_if_start(start_poses, char_pos):
    if char_pos[0] in start_poses:
        return True
    return False 


def get_word_position(start_poses, char_pos):
    return start_poses.index(char_pos[0])


def get_words(text, language):
    if language == "English":
        words = text.split()
    elif language == "Chinese":
        words = list(text)
    else:
        raise NotImplementedError
    return words


def get_left_and_right_pos(text, trigger, language):
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

