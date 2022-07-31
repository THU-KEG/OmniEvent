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


def generate_label2id_role2id(data_path: str):
    """Allocate an id for event types and roles within the event schema.

    Allocate an id for event types and roles within the event schema. The id of event types and roles start from 0.
    Finally, the correspondence of each event type/role and their id are stored in a dictionary, in which the key of
    each element is the event type/role, and the value is their id.

    Args:
        data_path: The path of the `duee_event_schema.json` schema file.
    """
    label2id, role2id = dict(NA=0), dict(NA=0)

    schema = list(jsonlines.open(data_path))
    for i, s in enumerate(schema):
        label2id[s["event_type"]] = i + 1
        for role in s["role_list"]:
            if role["role"] not in role2id:
                role2id[role["role"]] = len(role2id)

    io_dir = '/data/processed'.join("/".join(data_path.split("/")[:-2]).split('/data'))
    with open(os.path.join(io_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)

    with open(os.path.join(io_dir, "role2id.json"), "w", encoding="utf-8") as f:
        json.dump(role2id, f, indent=4, ensure_ascii=False)


def chinese_tokenizer(input_text: str,
                      tokenizer="jieba") -> List[str]:
    """Tokenize the Chinese input sequence into tokens.

    Tokenize the Chinese sequence into tokens by calling the relevant packages. The `chinese_tokenizer()` function
    integrates four commonly-used tokenizers, including Jieba, LTP, THULAC, and HanLP. The tokenized tokens are stored
    as a list for return.

    Args:
        input_text: The input text for tokenization.
        tokenizer: The tokenizer utilized for the tokenization process, such as Jieba, LTP, etc.

    Returns:
        A list of tokens after the tokenization of a Chinese input sequence. For example:

        ["雀巢", "裁员", "4000", "人", "：", "时代", "抛弃", "你", "时", "，", "连", "招呼", "都", "不会", "打", "！"]
    """
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


def re_tokenize(token_list: List[str],
                event: dict) -> List[str]:
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


def convert_duee_to_unified(data_path: str,
                            dump=True, tokenizer="jieba") -> list:
    """Convert DuEE 1.0 dataset to the unified format.

    Extract the information from the original DuEE 1.0 dataset and convert the format to a unified OpenEE dataset. The
    tokens not annotated as triggers are also regarded as negative triggers. The converted dataset is written to a json
    file.

    Args:
        data_path: The path of the original DuEE 1.0 dataset.
        dump: The setting of whether to write the manipulated dataset into a json file.
        tokenizer: The tokenizer utilized for the tokenization process, such as Jieba, LTP, etc.

    Returns:
        The manipulated dataset of DuEE 1.0 after converting its format into a unified OpenEE dataset. For example:

        {"id": "409389c96efe78d6af1c86e0450fd2d7", "text": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
         "events": [
            {"type": "组织关系-裁员",
             "triggers": [{"id": "409389c96efe78d6af1c86e0450fd2d7-17fc695a07a0ca6e0822e8f36c031199",
                           "trigger_word": "裁员", "position": [2, 4],
                           "arguments": [{"id": "bdd640fb06671ad11c80317fa3b1799d", "role": "裁员方", "mentions": [
                            {"mention_id": "23b8c1e9392456de3eb13b9046685257", "mention": "雀巢",
                             "position": [0, 2]}, ... ]}, ... ]},
                          ... ]}, ... ],
         "negative_triggers": [{"id": "409389c96efe78d6af1c86e0450fd2d7-9215ce3e90dd4507a282e59ca06a63eb",
                                "trigger_word": "雀巢", "position": [0, 2]}, ... ]}
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

            # manually remove bad case.
            if instance["id"] == "326ece324c848949f96db780db85fc22":
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
                    event["argument"].append({"id": str(uuid.UUID(int=random.getrandbits(128))).replace("-", ""),
                                              "role": role, "mentions": [{"mention_id": str(uuid.UUID(int=random.getrandbits(128))).replace("-", ""),
                                                                          "mention": arg["argument"],
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
                    "arguments": e["argument"]
                })

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

    data_path = '/data/processed'.join('/'.join(data_path.split('/')[:-1]).split('/data'))
    if dump:
        with jsonlines.open(data_path.replace(".json", ".unified.json"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    os.makedirs('../../../data/processed/DuEE1.0', exist_ok=True)
    generate_label2id_role2id("../../../data/DuEE1.0/duee_schema/duee_event_schema.json")
    convert_duee_to_unified("../../../data/DuEE1.0/duee_train.json/duee_train.json")
    convert_duee_to_unified("../../../data/DuEE1.0/duee_dev.json/duee_dev.json")
    convert_duee_to_unified("../../../data/DuEE1.0/duee_test2.json/duee_test2.json")
