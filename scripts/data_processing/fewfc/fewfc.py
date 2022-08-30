import argparse
import re
import os
import json
import uuid
import jieba
import random
import jsonlines

from tqdm import tqdm
from typing import List, Optional, Dict

random.seed(42)

label2id = {
    "NA": 0,
    "质押": 1,
    "股份股权转让": 2,
    "起诉": 3,
    "投资": 4,
    "减持": 5,
    "收购": 6,
    "担保": 7,
    "中标": 8,
    "签署合同": 9,
    "判决": 10
}


def get_role2id(train_file: str,
                test_file: str,
                output_file: str) -> None:
    """Generates the correspondence between argument roles and ids.

    Generates the correspondence between labels and ids, and roles and ids and saves them into a dictionary, in which
    the key of the dictionary is the argument role and the label is it's corresponding id.

    Args:
        train_file (`str`):
            A string indicating the path of the training dataset.
        test_file (`str`):
            A string indicating the path of the testing dataset.
        output_file (`str`):
            A string indicating the path for output the correspondence file.
    """
    role2id = dict(NA=0)
    data = list(jsonlines.open(train_file)) + list(jsonlines.open(test_file))
    for d in data:
        for event in d['events']:
            event_type = event['type']
            for mention in event['mentions']:
                role = mention['role']
                if role != 'trigger':
                    arg_role = '{}-{}'.format(event_type, role)
                    if arg_role not in role2id:
                        role2id[arg_role] = len(role2id)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(role2id, f, indent=4, ensure_ascii=False)


def split_training_data(train_file: str,
                        dev_file: str,
                        ratio: Optional[float] = 0.15) -> None:
    """Splits the training and validation set.

    Splits the training and validation set with a ratio of 0.85 and 0.15 randomly. 85% of the original dataset is
    regarded as the training set, while the rest of the 15% are regarded as the validation set.

    Args:
        train_file (`str`):
            A string indicating the path of the training dataset.
        dev_file (`str`):
            A string indicating the path of the validation dataset.
        ratio (`float`, `optional`, defaults to 0.15):
            A float indicating the ratio of the validation data.
    """
    train_data = list(jsonlines.open(train_file))
    random.shuffle(train_data)

    dev_data = train_data[0:int(ratio*len(train_data))]
    train_data = train_data[int(ratio*len(train_data)):]

    with jsonlines.open(train_file, 'w') as f:
        for t in train_data:
            jsonlines.Writer.write(f, t)

    with jsonlines.open(dev_file, 'w') as f:
        for d in dev_data:
            jsonlines.Writer.write(f, d)


def detect_nested(input_data: List[dict]) -> List[Dict]:
    """Detects the nested trigger annotations in FewFC.

    Some FewFC trigger and entity annotations are nested. The method detects and returns the nested annotations for
    further check and cleaning.

    Args:
        input_data (`List[dict]`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        nested_list (`List[Dict]`)
            A list of dictionaries containing the nested event triggers accompanied with their corresponding source
            texts within the FewFC datasets.
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


def chinese_tokenizer(input_text: str,
                      tokenizer="jieba") -> List[str]:
    """Tokenizes the Chinese input sequence into tokens.

    Tokenizes the Chinese sequence into tokens by calling the relevant packages. The `chinese_tokenizer()` method
    integrates four commonly-used tokenizers, including Jieba, LTP, THULAC, and HanLP. The tokenized tokens are stored
    as a list for return.

    Args:
        input_text (`str`):
            A string representing the input text for tokenization.
        tokenizer (`str`):
            A string indicating the tokenizer utilized for the tokenization process, such as Jieba, LTP, etc.

    Returns:
        token_list (`List[str]`):
            A list of strings representing the tokens after the tokenization of the Chinese input sequence.
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


def convert_fewfc_to_unified(data_path: str,
                             dump: Optional[bool] = True,
                             tokenizer: Optional[str] = "jieba") -> List[Dict]:
    """Converts FewFC dataset to the unified format.

    Extracts the information from the original FewFC dataset and convert the format to a unified OmniEvent dataset. The
    converted dataset is written to a json file.

    Args:
        data_path (`str`):
            A string indicating the path of the original FewFC dataset.
        dump (`bool`, `optional`, defaults to `True`):
            The setting of whether or not to write the manipulated dataset into a json file.
        tokenizer (`str`, `optional`, defaults to "jieba"):
            A string indicating the proposed Chinese tokenizer for the tokenization process.

    Returns:
        formatted_data (`List[Dict]`):
            A list of dictionaries representing the manipulated dataset of FewFC after converting its format into a
            unified OmniEvent dataset.
    """
    fewfc_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            fewfc_data.append(json.loads(line.strip().replace('�', ' ')))   # replace special chars

    formatted_data = []

    for sent_id, sent in enumerate(tqdm(fewfc_data)):
        instance = dict()

        instance["id"] = sent["id"]
        instance["text"] = sent["content"]

        tokens = chinese_tokenizer(instance["text"], tokenizer)

        # FewFC has provided labels for all data splits
        instance["events"] = []
        instance["negative_triggers"] = []
        events_in_sen = list()
        trigger_list = []

        # reformat and rank the events by length
        events = []
        for event in sent["events"]:
            tmp = dict(type=event["type"], trigger=str(), offset=list(), argument=list())
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
                    tmp["argument"].append({"id": str(uuid.UUID(int=random.getrandbits(128))).replace("-", ""),
                                            "role": "{}-{}".format(event["type"], d["role"]),
                                            "mentions": [{"mention_id": str(uuid.UUID(int=random.getrandbits(128))).replace("-", ""),
                                                          "mention": d["word"],
                                                          "position": d["span"]}]})
            if keep:
                events.append(tmp)
        events = sorted(events, key=lambda x: len(x["trigger"]))

        # re-tokenize the sentence according to the triggers
        for event in events:
            trigger_list.append(event["trigger"])
            events_in_sen.append(event)
            if event["trigger"] not in tokens:
                tokens = re_tokenize(tokens, event)

        for e in events_in_sen:
            event = dict()
            event["type"] = e["type"]
            event["triggers"] = []

            e["id"] = str(uuid.uuid4()).replace("-", "")

            event["triggers"].append({
                "id": "{}-{}".format(instance["id"], e["id"]),
                "trigger_word": e["trigger"],
                "position": e["offset"],
                "arguments": e["argument"]
            })

            assert instance["text"][e["offset"][0]: e["offset"][1]] == e["trigger"]

            for arg in e["argument"]:
                for a in arg["mentions"]:
                    assert instance["text"][a["position"][0]: a["position"][1]] == a["mention"]

            # the following triggers are not in the token list due to re-tokenizing.
            if e["trigger"] not in ["新增", "买入", "收购"]:
                assert e["trigger"] in tokens

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

    print("We get {}/{} instances for [{}].".format(len(formatted_data), len(fewfc_data), data_path))

    data_path = '/data/processed'.join(data_path.split('/data/original')).replace('_base', '')
    if dump:
        with jsonlines.open(data_path.replace(".json", ".unified.jsonl"), "w") as f:
            for item in formatted_data:
                jsonlines.Writer.write(f, item)

    return formatted_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="FewFC")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/original/FewFC")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/FewFC")
    args = arg_parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)

    get_role2id(train_file=os.path.join(args.data_dir, "train_base.json"),
                test_file=os.path.join(args.data_dir, "test_base.json"),
                output_file=os.path.join(args.save_dir, "role2id.json"))
    if not os.path.exists(os.path.join(args.data_dir, "valid_base.json")):
        split_training_data(train_file=os.path.join(args.data_dir, "train_base.json"),
                            dev_file=os.path.join(args.data_dir, "valid_base.json"))

    convert_fewfc_to_unified(os.path.join(args.data_dir, "train_base.json"))
    convert_fewfc_to_unified(os.path.join(args.data_dir, "valid_base.json"))
    convert_fewfc_to_unified(os.path.join(args.data_dir, "test_base.json"))
