import os 
import pdb 
import sys 
sys.path.append("..")

import argparse
import json 
from pathlib import Path 
from collections import defaultdict
from typing import List
from utils import generate_negative_trigger_per_item


def token_pos_to_char_pos(token_list: List[str],
                          start: int,
                          end: int,
                          mention: str) -> List[int]:
    """Converts the token-level position of a mention into character-level.

    Converts the token-level position of a mention into character-level by counting the number of characters before the
    mention's start and end positions.

    Args:
        token_list (`List[str]`):
            A list of strings representing the tokens within the source text.
        start (`int`):
            An integer indicating the word-level start position of the mention.
        end (`int`):
            An integer indicating the word-level end position of the mention.
        mention (`str`):
            A string representing the mention.

    Returns:
        `List[int]`:
            A list of integers representing the character-level start and end position of the mention.
    """
    char_start = len(" ".join(token_list[:start]))
    char_start += 0 if start == 0 else 1
    char_end = len(" ".join(token_list[:end]))
    if " ".join(token_list)[char_start:char_end] != mention:
        print("Warning!", " ".join(token_list)[char_start:char_end], "\tv.s.\t", mention)
    return [char_start, char_end]


def convert_to_openee(input_path: str,
                      save_path: str) -> None:
    """Convert ACE2005 OneIE dataset to the unified format.

    Extract the information from the original ACE2005 OneIE dataset and convert the format to a unified OpenEE dataset.
    The converted dataset is written to a json file.

    Args:
        input_path (`str`):
            A string indicating the path of the original ACE2005 OneIE dataset.
        save_path (`str`):
            A string indicating the saving path of the processed ACE2005 OneIE dataset.
    """
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    openee_data = []
    for item in data:
        openee_item = {
            "id": item["sent_id"],
            "text": " ".join(item["tokens"]),
            "events": [],
            "entities": []
        }
        # entities 
        entities_in_item = defaultdict(list)
        for entity in item["entity_mentions"]:
            entities_in_item[entity["id"]].append(entity)
            # append to openee
            mention = " ".join(item["tokens"][entity["start"]:entity["end"]])
            openee_entity = {
                "id": entity["id"],
                "type": entity["entity_type"],
                "mentions": [
                    {
                        "mention_id": "mention_0",
                        "mention": mention,
                        "position": token_pos_to_char_pos(item["tokens"], entity["start"], entity["end"], mention)
                    }
                ]
            }
            openee_item["entities"].append(openee_entity)
        # events 
        for event in item["event_mentions"]:
            openee_event = {
                "type": event["event_type"],
                "triggers": []
            }
            trigger_word = " ".join(item["tokens"][event["trigger"]["start"]:event["trigger"]["end"]])
            openee_trigger = {
                "id": "NA",
                "trigger_word": trigger_word,
                "position": token_pos_to_char_pos(item["tokens"], event["trigger"]["start"], event["trigger"]["end"], trigger_word),
                "arguments": []
            }
            assert trigger_word == openee_item["text"][openee_trigger["position"][0]:openee_trigger["position"][1]]
            for argument in event["arguments"]:
                argument_entity = None 
                for entity in entities_in_item[argument["entity_id"]]:
                    if entity["text"] == argument["text"]:
                        argument_entity = entity
                        break 
                assert argument_entity is not None 
                mention = " ".join(item["tokens"][argument_entity["start"]:argument_entity["end"]])
                openee_trigger["arguments"].append({
                    "role": argument["role"],
                    "mentions": [
                        {
                            "mention": mention,
                            "position": token_pos_to_char_pos(item["tokens"], argument_entity["start"], argument_entity["end"], mention)
                        }
                    ]
                })
            openee_event["triggers"].append(openee_trigger)
            openee_item["events"].append(openee_event)
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
    arg_parser = argparse.ArgumentParser(description="ACE2005-OneIE")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/ace2005-oneie")
    args = arg_parser.parse_args()

    dump_path = Path(args.save_dir)
    dump_path.mkdir(exist_ok=True, parents=True)
    for split in ["train", "dev", "test"]:
        save_split = split
        if split == "dev":
            save_split = "valid"
        convert_to_openee(f"data/{split}.oneie.json", os.path.join(dump_path, f"{save_split}.unified.jsonl"))
        generate_negative_trigger(os.path.join(dump_path, f"{save_split}.unified.jsonl"))
    get_ids(dump_path)

