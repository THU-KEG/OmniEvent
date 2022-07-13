import os 
import pdb 
import sys 
sys.path.append("..")
import json 
from pathlib import Path 
from collections import defaultdict
from utils import generate_negative_trigger_per_item


def token_pos_to_char_pos(token_list, start, end, mention):
    char_start = len(" ".join(token_list[:start]))
    char_start += 0 if start == 0 else 1
    char_end = len(" ".join(token_list[:end]))
    if " ".join(token_list)[char_start:char_end] != mention:
        print("Warning!", " ".join(token_list), mention)
    return [char_start, char_end]


def convert_to_openee(input_path, save_path):
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    openee_data = []
    for item in data:
        openee_item = {
            "id": item["sent_id"],
            "text": item["sentence"],
            "events": [],
            "entities": []
        }
        # entities 
        entities_in_item = defaultdict(list)
        for entity in item["entity_mentions"]:
            entities_in_item[entity["id"]].append(entity)
        # eventns 
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
            for argument in event["arguments"]:
                argument_entity = None 
                for entity in entities_in_item[argument["entity_id"]]:
                    if entity["text"] == argument["text"]:
                        argument_entity = entity
                        break 
                assert argument_entity is not None 
                openee_trigger["arguments"].append({
                    "role": argument["role"],
                    "mentions": [
                        {
                            "mention": argument_entity["text"],
                            "position": token_pos_to_char_pos(item["tokens"], argument_entity["start"], argument_entity["end"], argument_entity["text"])
                        }
                    ]
                })
            openee_event["triggers"].append(openee_trigger)
            openee_item["events"].append(openee_event)
        openee_data.append(openee_item)
    with open(save_path, "w") as f:
        for item in openee_data:
            f.write(json.dumps(item)+"\n")


def generate_negative_trigger(io_path):
    data = []
    with open(io_path) as f:
        for line in f.readlines():
            item = generate_negative_trigger_per_item(json.loads(line.strip()))
            data.append(item)
    with open(io_path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")


def get_ids(data_path):
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
    dump_path = Path("../../../data/processed/ace2005-oneie")
    dump_path.mkdir(exist_ok=True, parents=True)
    for split in ["train", "dev", "test"]:
        convert_to_openee(f"data/{split}.oneie.json", os.path.join(dump_path, f"{split}.unified.jsonl"))
        generate_negative_trigger(os.path.join(dump_path, f"{split}.unified.jsonl"))
    get_ids(dump_path)

