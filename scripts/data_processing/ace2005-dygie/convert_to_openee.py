import os
import pdb 
import sys
from typing import List

sys.path.append("..")
import json
import collections
from os import path
from pathlib import Path 
from utils import generate_negative_trigger_per_item


def convert_to_sentence():
    output_dir = "./default-settings"
    tmp_json_dir = "./default-settings"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for fold in ["train", "dev", "test"]:
        f_convert = open(path.join(output_dir, fold + "_convert.json"), "w")

        with open(path.join(tmp_json_dir, fold + ".json"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                json_obj = json.loads(line)

                sentences = json_obj["sentences"]
                ners = json_obj["ner"]
                relations = json_obj["relations"]
                events = json_obj["events"]
                sentence_start = json_obj["_sentence_start"]
                doc_key = json_obj["doc_key"]

                assert len(sentence_start) == len(ners) == len(relations) == len(events) == len(sentence_start)

                for sentence, ner, relation, event, s_start in zip(sentences, ners, relations, events, sentence_start):
                    # sentence_annotated = dict()
                    sentence_annotated = collections.OrderedDict()
                    sentence_annotated["sentence"] = sentence
                    sentence_annotated["s_start"] = s_start
                    sentence_annotated["ner"] = ner
                    # sentence_annotated["relation"] = relation
                    sentence_annotated["event"] = event

                    # if sentence_annotated["s_start"]>5:
                    f_convert.write(json.dumps(sentence_annotated, default=int) + "\n")


def token_pos_to_char_pos(token_list: List[str],
                          start: int,
                          mention: str) -> List[int]:
    """Converts the token-level position of a mention into character-level.

    Converts the token-level position of a mention into character-level by counting the number of characters before the
    start position of the mention. The end position could then be derived by adding the character-level start position
    and the length of the mention's span.

    Args:
        token_list (`List[str]`):
            A list of strings representing the tokens within the source text.
        start (`int`):
            An integer indicating the word-level position of the mention.
        mention (`str`):
            A string representing the mention.

    Returns:
        `List[int]`:
            A list of integers representing the character-level start and end position of the mention.
    """
    char_start = 0
    for i, token in enumerate(token_list):
        if i == start:
            break 
        char_start += len(token) + 1
    char_end = char_start + len(mention)
    assert " ".join(token_list)[char_start:char_end] == mention
    return [char_start, char_end]


def convert_to_openee(input_path: str,
                      save_path: str) -> None:
    """Convert ACE2005 DyGIE dataset to the unified format.

    Extract the information from the original ACE2005 DyGIE dataset and convert the format to a unified OpenEE dataset.
    The converted dataset is written to a json file.

    Args:
        input_path (`str`):
            A string indicating the path of the original ACE2005 DyGIE dataset.
        save_path (`str`):
            A string indicating the saving path of the processed ACE2005 DyGIE dataset.
    """
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    openee_data = []
    for item in data:
        openee_item = {
            "id": "NA",
            "text": " ".join(item["sentence"]),
            "events": [],
            "entities": []
        }
        for event in item["event"]:
            openee_event = {
                "type": "NA",
                "triggers": []
            }
            trigger = event[0]
            offset = trigger[0] - item["s_start"]
            trigger_word = item["sentence"][offset]
            openee_event["type"] = trigger[1]
            openee_trigger = {
                "id": "NA",
                "trigger_word": trigger_word,
                "position": token_pos_to_char_pos(item["sentence"], offset, trigger_word),
                "arguments": []
            }
            for argument in event[1:]:
                start = argument[0] - item["s_start"] 
                end = argument[1] - item["s_start"] + 1
                mention = " ".join(item["sentence"][start:end])
                role = argument[2]
                openee_trigger["arguments"].append({
                    "role": role,
                    "mentions": [
                        {
                            "mention": mention,
                            "position": token_pos_to_char_pos(item["sentence"], start, mention)
                        }
                    ]
                })
            openee_event["triggers"].append(openee_trigger)
            openee_item["events"].append(openee_event)
        for i, entity in enumerate(item["ner"]):
            start = entity[0] - item["s_start"]
            end = entity[1] - item["s_start"] + 1
            mention = " ".join(item["sentence"][start:end])
            openee_entity = {
                "id": f"entity-{i}",
                "type": entity[2],
                "mentions": [
                    {
                        "mention_id": "mention_0",
                        "mention": mention,
                        "position": token_pos_to_char_pos(item["sentence"], start, mention)
                    }
                ]
            }
            openee_item["entities"].append(openee_entity)
        openee_data.append(openee_item)
    with open(save_path, "w") as f:
        for item in openee_data:
            f.write(json.dumps(item)+"\n")


def generate_negative_trigger(io_path: str) -> None:
    """Generates negative triggers based on the triggers and source text.

    Generates negative triggers based on the triggers and source text. The tokens not within any trigger are regarded
    as negative triggers. The id, trigger word, and word-level position of each negative trigger are stored in a
    dictionary. The dictionaries are finally saved into a json file.

    Args:
        io_path (`str`):
            A string indicating the path to store the negative triggers file.
    """
    data = []
    with open(io_path) as f:
        for line in f.readlines():
            item = generate_negative_trigger_per_item(json.loads(line.strip()))
            data.append(item)
    with open(io_path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")


def get_ids(data_path: Path) -> None:
    """Generates the correspondence between labels and ids, and roles and ids.

    Generates the correspondence between labels and ids, and roles and ids, and save the correspondences into json
    files. Each label/role corresponds to a unique id.

    Args:
        data_path (`Dict`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.
    """
    data = []
    with open(os.path.join(data_path, "train.unified.jsonl")) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    role2id = {"NA": 0}
    label2id = {"NA": 0}
    for item in data:
        for event in item["events"]:
            if event["type"] not in label2id:
                label2id[event["type"]] = len(label2id)
            for trigger in event["triggers"]:
                for argument in trigger["arguments"]:
                    if argument["role"] not in role2id:
                        role2id[argument["role"]] = len(role2id)
    json.dump(label2id, open(os.path.join(data_path, "label2id.json"), "w"))
    json.dump(role2id, open(os.path.join(data_path, "role2id.json"), "w"))


if __name__ == "__main__":
    dump_path = Path("../../../data/processed/ace2005-dygie")
    dump_path.mkdir(parents=True, exist_ok=True)
    convert_to_sentence()
    for split in ["train", "dev", "test"]:
        out_file = f"{split}.unified.jsonl" if split != "dev" else "valid.unified.jsonl"   # use valid instead of dev
        convert_to_openee(f"default-settings/{split}_convert.json", os.path.join(dump_path, out_file))
        generate_negative_trigger(os.path.join(dump_path, out_file))
    get_ids(dump_path)
