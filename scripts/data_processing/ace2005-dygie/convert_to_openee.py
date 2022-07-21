import os
import pdb 
import sys 
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
                ner = json_obj["ner"]
                relations = json_obj["relations"]
                events = json_obj["events"]
                sentence_start = json_obj["_sentence_start"]
                doc_key = json_obj["doc_key"]

                assert len(sentence_start) == len(ner) == len(relations) == len(events) == len(sentence_start)

                for sentence, ner, relation, event, s_start in zip(sentences, ner, relations, events, sentence_start):
                    # sentence_annotated = dict()
                    sentence_annotated = collections.OrderedDict()
                    sentence_annotated["sentence"] = sentence
                    sentence_annotated["s_start"] = s_start
                    # sentence_annotated["ner"] = ner
                    # sentence_annotated["relation"] = relation
                    sentence_annotated["event"] = event

                    # if sentence_annotated["s_start"]>5:
                    f_convert.write(json.dumps(sentence_annotated, default=int) + "\n")


def token_pos_to_char_pos(token_list, start, mention):
    char_start = 0
    for i, token in enumerate(token_list):
        if i == start:
            break 
        char_start += len(token) + 1
    char_end = char_start + len(mention)
    assert " ".join(token_list)[char_start:char_end] == mention
    return [char_start, char_end]


def convert_to_openee(input_path, save_path):
    data = []
    with open(input_path) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    openee_data = []
    for item in data:
        openee_item = {
            "id": "NA",
            "text": " ".join(item["sentence"]),
            "events": []
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
    dump_path = Path("../../../data/processed/ace2005-dygie")
    dump_path.mkdir(parents=True, exist_ok=True)
    convert_to_sentence()
    for split in ["train", "dev", "test"]:
        convert_to_openee(f"default-settings/{split}_convert.json", os.path.join(dump_path, f"{split}.unified.jsonl"))
        generate_negative_trigger(os.path.join(dump_path, f"{split}.unified.jsonl"))
    get_ids(dump_path)
