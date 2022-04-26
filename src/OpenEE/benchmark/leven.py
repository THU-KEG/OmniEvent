import os
import json
import jsonlines

from tqdm import tqdm
from collections import defaultdict


def convert_leven_to_unified(data_path: str, dump=True) -> list:
    leven_data = list(jsonlines.open(data_path))

    label2id = dict(NA=0)
    formatted_data = []

    for item in tqdm(leven_data):
        for sent_id, sent in enumerate(item["content"]):
            instance = dict()

            instance["id"] = item["id"]
            instance["sentence"] = sent['sentence']

            if "events" not in item:
                # if test dataset, we don't have the labels.
                instance["candidates"] = []
                for candidate in item["candidates"]:
                    if candidate["sent_id"] == sent_id:
                        char_start = len("".join(sent["tokens"][:candidate["offset"][0]]))
                        char_end = char_start + \
                                   len("".join(sent["tokens"][candidate["offset"][0]:candidate["offset"][1]]))
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
                        char_start = len("".join(sent["tokens"][:mention["offset"][0]]))
                        char_end = char_start + len("".join(sent["tokens"][mention["offset"][0]:mention["offset"][1]]))

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
                        char_start = len("".join(sent["tokens"][:neg["offset"][0]]))
                        char_end = char_start + len("".join(sent["tokens"][neg["offset"][0]:neg["offset"][1]]))

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
        label2id = dict(sorted(list(label2id.items()), key=lambda x: x[1]))
        json.dump(label2id, open(os.path.join(io_dir, "label2id.json"), "w"), indent=4)

    if dump:
        with jsonlines.open(data_path.replace(".jsonl", ".unified.jsonl"), 'w') as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    convert_leven_to_unified("../../../data/LEVEN/train.jsonl")
    convert_leven_to_unified("../../../data/LEVEN/valid.jsonl")
    convert_leven_to_unified("../../../data/LEVEN/test.jsonl")
