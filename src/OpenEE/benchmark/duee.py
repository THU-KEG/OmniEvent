import os
import re
import json
import uuid
import jieba
import random
import jsonlines

from tqdm import tqdm
from typing import List

random.seed(42)


def generate_label2id(data_path: str):
    label2id = dict(NA=0)
    schema = list(jsonlines.open(data_path))
    for i, s in enumerate(schema):
        label2id[s["event_type"]] = i + 1

    io_dir = "/".join(data_path.split("/")[:-1])
    with open(os.path.join(io_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)


def chinese_tokenizer(input_text: str, tokenizer="jieba") -> List[str]:
    token_list = []
    if tokenizer == "jieba":
        token_list = jieba.lcut(input_text)

    # TODO: add other tokenizers
    if tokenizer == "ltp":
        raise NotImplementedError
    if tokenizer == "thulac":
        raise NotImplementedError
    if tokenizer == "hanlp":
        raise NotImplementedError

    return token_list


def re_tokenize(token_list: List[str], event: dict) -> List[str]:
    trigger = event["trigger"]
    offset = [event["trigger_start_index"], event["trigger_start_index"]+len(trigger)]

    start, brk1, brk2 = 0, -1, -1
    for i, t in enumerate(token_list):
        if brk1 == -1:
            if start <= offset[0] < start + len(t):
                brk1 = i
                if offset[1] <= start + len(t):
                    brk2 = i + 1
                    break
        else:
            if start <= offset[1] <= start + len(t):
                brk2 = i + 1
                break
        start += len(t)

    left = token_list[:brk1]
    middle = token_list[brk1:brk2]
    right = token_list[brk2:]

    middle_new = [token for token in re.split("(" + str(trigger) + ")", "".join(middle)) if token]

    return left + middle_new + right


def convert_duee_to_unified(data_path: str, dump=True, tokenizer="jieba") -> list:
    """
    Convert DuEE1.0 dataset to the unified format.
    Dataset link: https://www.luge.ai/#/luge/dataDetail?id=6
    """
    duee_data = list(jsonlines.open(data_path))

    formatted_data = []
    error_annotations = []

    for sent_id, sent in enumerate(tqdm(duee_data)):
        instance = dict()

        instance["id"] = sent["id"]
        instance["text"] = sent["text"]

        tokens = chinese_tokenizer(sent["text"], tokenizer)

        if "test" in data_path:
            # if test dataset, we don't have the labels.
            instance["candidates"] = []
            start = 0
            for candidate in tokens:
                char_start = start
                char_end = char_start + len(candidate)
                start = char_end

                instance["candidates"].append({
                    "id": "{}-{}".format(instance["id"], str(uuid.UUID(int=random.getrandbits(128))).replace("-", "")),
                    "trigger_word": candidate,
                    "position": [char_start, char_end]
                })
                assert instance["text"][char_start:char_end] == candidate
        else:
            if "event_list" not in sent:
                error_annotations.append(sent)
                continue

            # if train dataset, we have the labels.
            instance["events"] = list()
            instance["negative_triggers"] = list()
            events_in_sen = list()

            trigger_list = []
            for event in sent["event_list"]:
                event["argument"] = list()
                for arg in event["arguments"]:
                    role = arg["role"]
                    arg_start = arg["argument_start_index"]
                    arg_end = arg_start + len(arg["argument"])
                    event["argument"].append({"role": role, "mentions": [{"mention": arg["argument"],
                                                                          "position": [arg_start, arg_end]}]})
                events_in_sen.append(event)
                trigger_list.append(event["trigger"])
                if event["trigger"] not in tokens:
                    tokens = re_tokenize(tokens, event)

            for e in events_in_sen:
                event = dict()
                event["type"] = e["event_type"]
                event["triggers"] = []

                char_start = e["trigger_start_index"]
                char_end = char_start + len(e["trigger"])
                e["id"] = str(uuid.UUID(int=random.getrandbits(128))).replace("-", "")

                event["triggers"].append({
                    "id": "{}-{}".format(instance["id"], e["id"]),
                    "trigger_word": e["trigger"],
                    "position": [char_start, char_end],
                })

                event["arguments"] = e['argument']
                assert instance["text"][char_start:char_end] == e["trigger"]
                assert e["trigger"] in tokens
                for arg in e["argument"]:
                    for a in arg["mentions"]:
                        assert instance["text"][a["position"][0]: a["position"][1]] == a["mention"]

                instance["events"].append(event)

            # negative triggers
            start = 0
            for token in tokens:
                char_start = start
                char_end = char_start + len(token)
                start = char_end

                if token not in trigger_list:
                    negative = token
                    instance["negative_triggers"].append({
                        "id": "{}-{}".format(instance["id"], str(uuid.uuid4()).replace("-", "")),
                        "trigger_word": negative,
                        "position": [char_start, char_end]
                    })
                    assert instance["text"][char_start:char_end] == negative
        formatted_data.append(instance)

    print("We get {}/{} instances for [{}].".format(len(formatted_data), len(duee_data), data_path))

    if dump:
        with jsonlines.open(data_path.replace(".json", ".unified.json"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    generate_label2id("../../../data/DuEE1.0/duee_event_schema.json")
    convert_duee_to_unified("../../../data/DuEE1.0/duee_train.json")
    convert_duee_to_unified("../../../data/DuEE1.0/duee_dev.json")
    convert_duee_to_unified("../../../data/DuEE1.0/duee_test2.json")
