import argparse
import os
import json
import jsonlines
from tqdm import tqdm
from typing import Dict, List


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


def convert_leven_to_unified(data_path: str,
                             save_path: str,
                             dump=True) -> List[Dict]:
    """Convert LEVEN dataset to the unified format.

    Extract the information from the original LEVEN dataset and convert the format to a unified OmniEvent dataset. The
    converted dataset is written to a json file.

    Args:
        data_path (`str`):
            A string indicating the path of the original LEVEN dataset.
        save_path (`str`):
            A string indicating the path to save the unified LEVEN dataset.
        dump (`bool`, `optional`, defaults to `True`):
            A boolean variable indicating whether or not writing the manipulated dataset to a json file.

    Returns:
        formatted_data (`List[Dict]`):
            A list of dictionaries representing the manipulated dataset of LEVEN after converting its format into a
            unified OmniEvent dataset.
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

    # Change full punctuations into half.
    for line in formatted_data:
        line["text"] = str_full_to_half(line["text"])
        if "events" in line.keys():
            for event in line["events"]:
                for trigger in event["triggers"]:
                    trigger["trigger_word"] = str_full_to_half(trigger["trigger_word"])
            for trigger in line["negative_triggers"]:
                trigger["trigger_word"] = str_full_to_half(trigger["trigger_word"])
        else:
            for trigger in line["candidates"]:
                trigger["trigger_word"] = str_full_to_half(trigger["trigger_word"])

    if "train" in data_path:
        label2id = dict(sorted(list(label2id.items()), key=lambda x: x[1]))
        json.dump(label2id, open(os.path.join(save_path, "label2id.json"), "w", encoding='utf-8'),
                  indent=4, ensure_ascii=False)

    data_path = '/data/processed'.join(data_path.split('/data/original'))
    if dump:
        with jsonlines.open(data_path.replace(".jsonl", ".unified.jsonl"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="LEVEN")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/original/LEVEN")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/LEVEN")
    args = arg_parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    convert_leven_to_unified(os.path.join(args.data_dir, "train.jsonl"), args.save_dir)
    convert_leven_to_unified(os.path.join(args.data_dir, "valid.jsonl"), args.save_dir)
    convert_leven_to_unified(os.path.join(args.data_dir, "test.jsonl"), args.save_dir)
