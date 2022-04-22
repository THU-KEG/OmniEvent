import os 

from typing import Dict


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