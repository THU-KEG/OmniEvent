import os
import re
import json
import uuid
import jieba
import jsonlines

from tqdm import tqdm
from typing import List
from collections import defaultdict


label2id = {
    "NA": 0,
    "质押": 1,
    "股份股权转让": 2,
    "投资": 3,
    "起诉": 4,
    "高管减持": 5,
    "收购": 6,
    "判决": 7,
    "担保": 8,
    "中标": 9,
    "签署合同": 10
}


def detect_nested(input_data: List[dict]) -> List[dict]:
    """
    Detect the nested trigger annotations in FewFC dataset in order to manually clean those wrong annotations
    """
    nested_list = []
    for sent in input_data:
        offsets = []
        nested = []
        for event in sent["events"]:
            for d in event["mentions"]:
                if d["role"] == "trigger":
                    event["trigger"] = d["word"]
                    event["offset"] = d["span"]
                    for j, oft in enumerate(offsets):
                        if oft != d["span"] and \
                                set(range(oft[0], oft[1])).intersection(set(range(d["span"][0], d["span"][1]))):
                            nested.append(sent["events"][j])
                            nested.append(event)

                    offsets.append(d["span"])
        if nested:
            nested = {"sent": sent["content"], "nested": nested}
            nested_list.append(nested)
    return nested_list


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
    offset = event["offset"]

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


def convert_fewfc_to_unified(data_path: str, dump=True, tokenizer="jieba") -> list:
    """
    Convert FewFC dataset to unified format.
    Dataset link: https://github.com/TimeBurningFish/FewFC/tree/main/rearranged
    """
    fewfc_data = list(jsonlines.open(data_path))

    formatted_data = []

    for sent_id, sent in enumerate(tqdm(fewfc_data)):
        instance = dict()

        instance["id"] = sent["id"]
        instance["sentence"] = sent["content"]

        tokens = chinese_tokenizer(sent["content"], tokenizer)

        if "test" in data_path:
            # if test dataset, we have to provide candidates
            instance["candidates"] = []
            start = 0
            for candidate in tokens:
                char_start = start
                char_end = char_start + len(candidate)
                start = char_end

                instance["candidates"].append({
                    "id": "{}-{}".format(instance["id"], str(uuid.uuid4()).replace("-", "")),
                    "trigger_word": candidate,
                    "position": [char_start, char_end]
                })
                assert instance["sentence"][char_start:char_end] == candidate
        else:
            # if train dataset, we have the labels.
            instance["events"] = []
            instance["negative_triggers"] = []
            events_in_sen = defaultdict(list)
            trigger_list = []

            # reformat and rank the events by length
            events = []
            for event in sent["events"]:
                tmp = dict(type=event["type"], trigger=str(), offset=list(), argument=defaultdict(list))
                keep = True
                for d in event["mentions"]:
                    if d["role"] == "trigger":
                        # manually fix the annotation boundary errors in the dataset
                        if d["word"] in ["了收", "的增", "讼诉", "元取", "员减"] \
                                and d["span"] in [[14, 16], [70, 72], [18, 20], [50, 52], [20, 22]]:
                            keep = False
                            break
                        tmp["trigger"] = d["word"]
                        tmp["offset"] = d["span"]
                    else:
                        tmp["argument"][d["role"]].append({"mention": d["word"], "position": d["span"]})
                if keep:
                    events.append(tmp)
            events = sorted(events, key=lambda x: len(x["trigger"]))

            # re-tokenize the sentence according to the triggers
            for event in events:
                trigger_list.append(event["trigger"])
                events_in_sen[event["type"]].append(event)
                if event["trigger"] not in tokens:
                    tokens = re_tokenize(tokens, event)

            for type in events_in_sen:
                event = dict()
                event["type"] = type
                event["mentions"] = []
                for mention in events_in_sen[type]:
                    mention["id"] = str(uuid.uuid4()).replace("-", "")

                    event["mentions"].append({
                        "id": "{}-{}".format(instance["id"], mention["id"]),
                        "trigger_word": mention["trigger"],
                        "position": mention["offset"],
                        "argument": mention["argument"],
                    })

                    assert instance["sentence"][mention["offset"][0]: mention["offset"][1]] == mention["trigger"]
                    for arg in mention["argument"].values():
                        for a in arg:
                            assert instance["sentence"][a["position"][0]: a["position"][1]] == a["mention"]

                    # the following triggers are not in the token list due to re-tokenizing.
                    if mention["trigger"] not in ["新增", "买入", "收购"]:
                        assert mention["trigger"] in tokens

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
                    assert instance["sentence"][char_start:char_end] == negative
        formatted_data.append(instance)

    print("We get {}/{} instances for [{}].".format(len(formatted_data), len(fewfc_data), data_path))

    if dump:
        with jsonlines.open(data_path.replace(".json", ".unified.json"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    json.dump(label2id, open("../../../data/FewFC/label2id.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    convert_fewfc_to_unified("../../../data/FewFC/train_base.json")
    convert_fewfc_to_unified("../../../data/FewFC/train_trans.json")
    convert_fewfc_to_unified("../../../data/FewFC/test_base.json")
    convert_fewfc_to_unified("../../../data/FewFC/test_trans.json")
