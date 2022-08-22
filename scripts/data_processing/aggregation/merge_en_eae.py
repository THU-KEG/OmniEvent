import os 
import json
import random 
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
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")
<<<<<<< HEAD


def merge(data_dir: str) -> None:
    """Merges the processed event argument extraction datasets into larger datasets.

    Merges the processed event argument extraction datasets into larger datasets. The merged training dataset includes
    the pieces from LDC2015E29, LDC2015E68, LDC2015E78, and TAC KBP 2016, and the merged testing dataset includes the
    ACE2005-DyGIE and TAC KBP 2017 dataset. The merged datasets are stored into jsonl files.

    Args:
        data_dir (`str`):
            A string indicating the directory to place the written jsonl file.
    """
    all_train = []
    with open(os.path.join(data_dir, "ace2005-dygie/train.unified.jsonl")) as f:
=======
    

def load_jsonl(path, prefix):
    data = []
    with open(path) as f:
>>>>>>> dev
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
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2016/pilot.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "TAC-KBP2016/test.unified.jsonl"), "<kbp>")
    all_train += load_jsonl(os.path.join(data_dir, "DuEE1.0/duee_train.unified.json"), "<duee>")
    all_train += load_jsonl(os.path.join(data_dir, "FewFC/train_base.unified.json"), "<fewfc>")
    # dev
    all_dev = []
    all_dev += load_jsonl(os.path.join(data_dir, "ace2005-dygie/dev.unified.jsonl"), "<ace>")
    all_dev += load_jsonl(os.path.join(data_dir, "DuEE1.0/duee_dev.unified.json"), "<duee>")
    all_dev += load_jsonl(os.path.join(data_dir, "FewFC/dev_base.unified.json"), "<fewfc>")
    # test 
    all_test = []
    all_test += load_jsonl(os.path.join(data_dir, "ace2005-dygie/test.unified.jsonl"), "<ace>")
    all_test += load_jsonl(os.path.join(data_dir, "DuEE1.0/duee_dev.unified.json"), "<duee>")
    all_test += load_jsonl(os.path.join(data_dir, "FewFC/test_base.unified.json"), "<fewfc>")
    # save 
    print("All train: %d, all dev: %d, all test: %d" % (len(all_train), len(all_dev), len(all_test)))
    save_dir = Path("../../../data/processed/all-eae")
    save_dir.mkdir(exist_ok=True)
    save_jsonl(all_train, os.path.join(save_dir, "train.unified.jsonl"))
    save_jsonl(all_dev, os.path.join(save_dir, "dev.unified.jsonl"))
    save_jsonl(all_test, os.path.join(save_dir, "test.unified.jsonl"))


if __name__ == "__main__":
    merge("../../../data/processed")





