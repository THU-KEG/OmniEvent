import os
import json
import jsonlines
from tqdm import tqdm


def convert_leven_to_unified(data_path: str,
                             dump=True) -> list:
    """Convert LEVEN dataset to the unified format.

    Extract the information from the original LEVEN dataset and convert the format to a unified OpenEE dataset. The
    converted dataset is written to a json file.

    Args:
        data_path: The path of the original LEVEN dataset.
        dump: The setting of whether to write the manipulated dataset into a json file.

    Returns:
        The manipulated dataset of LEVEN after converting its format into a unified OpenEE dataset. For example:

        {"id": "c6663b3c88ed4826a4b1b22b1ef8370a", "text": "2011年7月至2012年7月间，被告人王师才为非法获利，...",
         "events": [
            {"type": "获利",
             "triggers": [{"id": "c6663b3c88ed4826a4b1b22b1ef8370a-c42390c4d1cb4b65a8329199c639ded0",
                           "trigger_word": "获利", "position": [26, 28]}, ... ]
         "negative_triggers": [{"id": "c6663b3c88ed4826a4b1b22b1ef8370a-f44294881cc34f8fb5c75dc98367e174",
                                "trigger_word": "月间", "position": [14, 16]}, ... ]}
    """
    leven_data = list(jsonlines.open(data_path))

    label2id = dict(NA=0)
    formatted_data = []

    for item in tqdm(leven_data):
        for sent_id, sent in enumerate(item["content"]):
            instance = dict()

            instance["id"] = item["id"]
            instance["text"] = sent['sentence']

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
                        assert instance["text"][char_start:char_end] == candidate["trigger_word"]
            else:
                # if train dataset, we have the labels.
                instance["events"] = list()
                instance["negative_triggers"] = list()
                events_in_sen = list()

                for event in item["events"]:
                    label2id[event["type"]] = event["type_id"]
                    for mention in event["mention"]:
                        if mention["sent_id"] == sent_id:
                            events_in_sen.append(dict(type=event["type"], mention=mention))

                for e in events_in_sen:
                    mention = e["mention"]
                    char_start = len("".join(sent["tokens"][:mention["offset"][0]]))
                    char_end = char_start + len("".join(sent["tokens"][mention["offset"][0]:mention["offset"][1]]))

                    event = dict()
                    event["type"] = e['type']

                    trigger = dict()
                    trigger['id'] = "{}-{}".format(instance["id"], mention["id"])
                    trigger["trigger_word"] = mention["trigger_word"]
                    trigger["position"] = [char_start, char_end]

                    event['triggers'] = [trigger]
                    assert instance["text"][char_start:char_end] == trigger["trigger_word"]

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
                        assert instance["text"][char_start:char_end] == neg["trigger_word"]
            formatted_data.append(instance)

    print("We get {} instances for [{}].".format(len(formatted_data), data_path))
    if "train" in data_path:
        io_dir = '/data/processed'.join("/".join(data_path.split("/")[:-1]).split('/data'))
        label2id = dict(sorted(list(label2id.items()), key=lambda x: x[1]))
        json.dump(label2id, open(os.path.join(io_dir, "label2id.json"), "w", encoding='utf-8'),
                  indent=4, ensure_ascii=False)

    data_path = '/data/processed'.join(data_path.split('/data'))
    if dump:
        with jsonlines.open(data_path.replace(".jsonl", ".unified.jsonl"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    os.makedirs("../../../data/processed/LEVEN/", exist_ok=True)
    convert_leven_to_unified("../../../data/LEVEN/train.jsonl")
    convert_leven_to_unified("../../../data/LEVEN/valid.jsonl")
    convert_leven_to_unified("../../../data/LEVEN/test.jsonl")
