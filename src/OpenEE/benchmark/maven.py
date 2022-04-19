import enum
import os 
import pdb 
import json 

from tqdm import tqdm 
from collections import defaultdict


def convert_maven_to_unified(data_path: str, dump=True) -> dict:
    """
    Convert the maven dataset to unified format.
    """
    # Load the maven dataset.
    maven_data = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            maven_data.append(json.loads(line.strip()))
    label2id = dict(NA=0)
    formatted_data = []
    for item in tqdm(maven_data):
        for sent_id, sen in enumerate(item["content"]):
            instance = dict()
            instance["id"] = item["id"]
            instance["sentence"] = " ".join(sen["tokens"]) 
            if "events" not in item: 
                # if test dataset, we don't have the labels.
                instance["candidates"] = []
                for candidate in item["candidates"]:
                    if candidate["sent_id"] == sent_id:
                        char_start = len(" ".join(sen["tokens"][:candidate["offset"][0]]))
                        # white space
                        if candidate["offset"][0] != 0:
                            char_start += 1
                        char_end = char_start + \
                            len(" ".join(sen["tokens"][candidate["offset"][0]:candidate["offset"][1]]))
                        instance["candidates"].append({
                            "id": "{}-{}".format(instance["id"], candidate["id"]),
                            "trigger_word": candidate["trigger_word"],
                            "position": [char_start, char_end]
                        })
                        assert instance["sentence"][char_start:char_end] == candidate["trigger_word"]
            else:
                # if train dataset, we have the labels.
                instance["events"] = []
                instance["negative_triggers"] = []
                events_in_sen = defaultdict(list)
                for event in item["events"]:
                    label2id[event["type"]] = event["type_id"]
                    for mention in event["mention"]:
                        if mention["sent_id"] == sent_id:
                            events_in_sen[event["type"]].append(mention)
                for type in events_in_sen:
                    event = dict()
                    event["type"] = type 
                    event["mentions"] = []
                    for mention in events_in_sen[type]:
                        char_start = len(" ".join(sen["tokens"][:mention["offset"][0]]))
                        # white space
                        if mention["offset"][0] != 0:
                            char_start += 1
                        char_end = char_start + \
                            len(" ".join(sen["tokens"][mention["offset"][0]:mention["offset"][1]]))
                        event["mentions"].append({
                            "id": "{}-{}".format(instance["id"], mention["id"]),
                            "trigger_word": mention["trigger_word"],
                            "position": [char_start, char_end]
                        })
                        assert instance["sentence"][char_start:char_end] == mention["trigger_word"]
                    instance["events"].append(event)
                # negative triggers 
                for neg in item["negative_triggers"]:
                    if neg["sent_id"] == sent_id:
                        char_start = len(" ".join(sen["tokens"][:neg["offset"][0]]))
                        # white space
                        if neg["offset"][0] != 0:
                            char_start += 1
                        char_end = char_start + \
                            len(" ".join(sen["tokens"][neg["offset"][0]:neg["offset"][1]]))
                        instance["negative_triggers"].append({
                            "id": "{}-{}".format(instance["id"], neg["id"]),
                            "trigger_word": neg["trigger_word"],
                            "position": [char_start, char_end]
                        })
                        assert instance["sentence"][char_start:char_end] == neg["trigger_word"]
            formatted_data.append(instance)
    print("We get {} instances.".format(len(formatted_data)))
    if "train" in data_path:
        io_dir = "/".join(data_path.split("/")[:-1])
        json.dump(label2id, open(os.path.join(io_dir, "label2id.json"), "w"), indent=4)
    if dump:
        with open(data_path.replace(".jsonl", ".unified.jsonl"), 'w') as f:
            for item in formatted_data:
                f.write(json.dumps(item)+"\n")
    return formatted_data


if __name__ == "__main__":
    convert_maven_to_unified("../../../data/MAVEN/train.jsonl")
    convert_maven_to_unified("../../../data/MAVEN/valid.jsonl")
    convert_maven_to_unified("../../../data/MAVEN/test.jsonl")
