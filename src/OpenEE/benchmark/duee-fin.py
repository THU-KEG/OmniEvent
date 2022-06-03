import os
import re
import json
import uuid
import jieba
import random
import jsonlines

from tqdm import tqdm
from typing import List
from collections import defaultdict

random.seed(42)


def generate_label2id_role2id(data_path: str):
    label2id, role2id = dict(NA=0), dict(NA=0)

    schema = list(jsonlines.open(data_path))
    for i, s in enumerate(schema):
        label2id[s["event_type"]] = i + 1
        for role in s["role_list"]:
            if role["role"] not in role2id:
                role2id[role["role"]] = len(role2id)

    io_dir = "/".join(data_path.split("/")[:-1])
    with open(os.path.join(io_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)

    with open(os.path.join(io_dir, "role2id.json"), "w", encoding="utf-8") as f:
        json.dump(role2id, f, indent=4, ensure_ascii=False)


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


def re_tokenize(token_list: List[str], mention: dict) -> List[str]:
    trigger = mention["trigger"]
    offset = [mention["trigger_start_index"], mention["trigger_start_index"]+len(trigger)]

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

    middle_new = [token for token in re.split("("+str(trigger)+")", "".join(middle)) if token]

    return left + middle_new + right


def convert_dueefin_to_unified(data_path: str, dump=True, tokenizer="jieba") -> list:
    """
    Convert DuEE-fin dataset to unified format.
    Dataset link: https://www.luge.ai/#/luge/dataDetail?id=7
    """
    dueefin_data = list(jsonlines.open(data_path))

    formatted_data = []
    error_annotations = []

    for sent_id, sent in enumerate(tqdm(dueefin_data)):
        instance = dict()

        instance["id"] = sent["id"]
        instance["text"] = " ".join([sent["title"], sent["text"]])  # concatenate the title and text

        tokens = chinese_tokenizer(instance["text"], tokenizer)

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
                    "position": [char_start, char_end],
                })
                assert instance["text"][char_start:char_end] == candidate
        else:
            # if train dataset, we have the labels.
            if "event_list" not in sent:
                error_annotations.append(sent)
                continue

            instance["events"] = list()
            instance["negative_triggers"] = list()
            events_in_sen = list()

            trigger_list = []
            trigger_offsets = []

            # rank the event list to localize the long triggers first.
            sent["event_list"] = sorted(sent["event_list"], key=lambda x: len(x["trigger"]), reverse=True)
            for event in sent["event_list"]:
                if event["trigger"] not in sent["text"]:
                    continue

                # manually fix the annotation boundary error in the dataset
                if event["trigger"][0] == ")":
                    event["trigger"] = event["trigger"][1:]

                # find the start index of the trigger in the sentence
                index = None
                if event["trigger"] in trigger_list:
                    index = trigger_offsets[trigger_list.index(event["trigger"])][0]
                else:
                    # find all the positions where the trigger appear in the sentence.
                    indices = [s.start() for s in re.finditer(event["trigger"], instance["text"])]
                    # use the first position, which doesn't collide with other longer triggers, as the start index.
                    for idx in indices:
                        keep = True
                        for offset in trigger_offsets:
                            if idx in range(offset[0], offset[1]):
                                keep = False
                                break
                        if keep:
                            index = idx
                            break

                event["trigger_start_index"] = index
                trigger_list.append(event["trigger"])
                trigger_offsets.append([index, index + len(event["trigger"])])
                if event["trigger"] not in tokens:
                    tokens = re_tokenize(tokens, event)

                # argument
                event["argument"] = list()
                for arg in event["arguments"]:
                    arg_start = instance["text"].find(arg["argument"])
                    arg_end = arg_start + len(arg["argument"]) if arg_start != -1 else -1

                    if arg_start == arg_end == -1 and arg["role"] != "环节":
                        continue

                    event["argument"].append({"id": str(uuid.UUID(int=random.getrandbits(128))).replace("-", ""),
                                              "role": arg["role"], "mentions": [{"mention_id": str(uuid.UUID(int=random.getrandbits(128))).replace("-", ""),
                                                                                 "mention": arg["argument"],
                                                                                 "position": [arg_start, arg_end]}]})

                events_in_sen.append(event)

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
                    "arguments": e["argument"]
                })

                assert instance["text"][char_start:char_end] == e["trigger"]
                assert e["trigger"] in tokens

                for arg in e["argument"]:
                    if arg["role"] != "环节":
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
                        "id": "{}-{}".format(instance["id"],
                                             str(uuid.UUID(int=random.getrandbits(128))).replace("-", "")),
                        "trigger_word": negative,
                        "position": [char_start, char_end]
                    })
                    assert instance["text"][char_start:char_end] == negative
        formatted_data.append(instance)

    print("We get {}/{} instances for [{}].".format(len(formatted_data), len(dueefin_data), data_path))

    if dump:
        with jsonlines.open(data_path.replace(".json", ".unified.json"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    generate_label2id_role2id("../../../data/DuEE-fin/duee_fin_event_schema.json")
    convert_dueefin_to_unified("../../../data/DuEE-fin/duee_fin_train.json")
    convert_dueefin_to_unified("../../../data/DuEE-fin/duee_fin_dev.json")
    convert_dueefin_to_unified("../../../data/DuEE-fin/duee_fin_test2.json")
