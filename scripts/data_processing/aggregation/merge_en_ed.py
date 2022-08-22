import os 
import json
import random

<<<<<<< HEAD

def save_jsonl(data, path):
=======
from pathlib import Path
from typing import Dict, List


def save_jsonl(data: List[Dict],
               path: str) -> None:
    """Write the manipulated dataset into a jsonl file.

    Write the manipulated dataset into a jsonl file; each line of the jsonl file corresponds to a piece of data.

    Args:
        data (`Dic`):
            A list of dictionaries indicating the manipulated dataset.
        path (`str`):
            A string indicating the path to place the written jsonl file.
    """
>>>>>>> b80759bfb01ea838588244fff858dea131792ad2
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")
    

<<<<<<< HEAD
def load_jsonl(path, prefix):
    data = []
    with open(path) as f:
=======
def merge(data_dir: str) -> None:
    """Merges the processed event detection datasets into larger datasets.

    Merges the processed event detection datasets into larger datasets. The merged training dataset includes the pieces
    from LDC2015E29, LDC2015E68, LDC2015E78, TAC KBP 2014, TAC KBP 2015, and TAC KBP 2016; the merged validation dataset
    includes ACE2005-DyGIE and MAVEN; the testing dataset is the TAC KBP 2017 dataset. The merged datasets are stored
    into jsonl files.

    Args:
        data_dir (`str`):
            A string indicating the directory to place the written jsonl file.
    """
    all_train = []
    ere = []
    with open(os.path.join(data_dir, "ace2005-dygie/train.unified.jsonl")) as f:
        for line in f.readlines():
            ere.append(json.loads(line.strip()))
    ere += json.load(open(os.path.join(data_dir, "ere/LDC2015E29.json")))
    ere += json.load(open(os.path.join(data_dir, "ere/LDC2015E68.json")))
    ere += json.load(open(os.path.join(data_dir, "ere/LDC2015E78.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2014/train.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2014/test.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2015/train.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2015/test.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2016/pilot.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2016/test.json")))
    for item in ere:
        item["source"] = "<ere>"
        all_train.append(item)
    with open(os.path.join(data_dir, "MAVEN/train.unified.jsonl")) as f:
>>>>>>> b80759bfb01ea838588244fff858dea131792ad2
        for line in f.readlines():
            item = json.loads(line.strip())
            item["source"] = prefix 
            data.append(item)
    return data


def merge(data_dir):
    # train
    all_train = []
    all_train += load_jsonl(os.path.join(data_dir, "ace2005-dygie/train.unified.jsonl"), "<ace>")
    all_train += load_jsonl(os.path.join(data_dir, "ere/LDC2015E29.unified.jsonl"), "<ere>")
    all_train += load_jsonl(os.path.join(data_dir, "ere/LDC2015E68.unified.jsonl"), "<ere>")
    all_train += load_jsonl(os.path.join(data_dir, "ere/LDC2015E78.unified.jsonl"), "<ere>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2014/train.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2014/test.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2015/train.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2015/test.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2016/pilot.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2016/test.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "MAVEN/train.unified.jsonl"), "<maven>")
    all_train += load_jsonl(os.path.join(data_dir, "DuEE1.0/duee_train.unified.json"), "<duee>")
    all_train += load_jsonl(os.path.join(data_dir, "FewFC/train_base.unified.json"), "<fewfc>")
    all_train += load_jsonl(os.path.join(data_dir, "LEVEN/train.unified.jsonl"), "<leven>")
    # dev
    all_dev = []
    all_dev += load_jsonl(os.path.join(data_dir, "ace2005-dygie/dev.unified.jsonl"), "<ace>")
    all_dev += load_jsonl(os.path.join(data_dir, "MAVEN/valid.unified.jsonl"), "<maven>")
    all_dev += load_jsonl(os.path.join(data_dir, "DuEE1.0/duee_dev.unified.json"), "<duee>")
    all_dev += load_jsonl(os.path.join(data_dir, "FewFC/dev_base.unified.json"), "<fewfc>")
    all_dev += load_jsonl(os.path.join(data_dir, "LEVEN/valid.unified.jsonl"), "<leven>")
    # test 
    all_test = []
    all_test += load_jsonl(os.path.join(data_dir, "ace2005-dygie/test.unified.jsonl"), "<ace>")
    all_test += load_jsonl(os.path.join(data_dir, "MAVEN/valid.unified.jsonl"), "<maven>")
    all_test += load_jsonl(os.path.join(data_dir, "DuEE1.0/duee_dev.unified.json"), "<duee>")
    all_test += load_jsonl(os.path.join(data_dir, "FewFC/test_base.unified.json"), "<fewfc>")
    all_test += load_jsonl(os.path.join(data_dir, "LEVEN/valid.unified.jsonl"), "<leven>")
    # save 
    print("All train: %d, all dev: %d, all test: %d" % (len(all_train), len(all_dev), len(all_test)))
    save_dir = Path("../../../data/processed/all-ed")
    save_dir.mkdir(exist_ok=True)
    save_jsonl(all_train, os.path.join(save_dir, "train.unified.jsonl"))
    save_jsonl(all_dev, os.path.join(save_dir, "dev.unified.jsonl"))
    save_jsonl(all_test, os.path.join(save_dir, "test.unified.jsonl"))


if __name__ == "__main__":
    merge("../../../data/processed")





