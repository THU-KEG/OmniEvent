import os 
import json
import random 
from pathlib import Path 


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")
    

def load_jsonl(path, prefix):
    data = []
    with open(path) as f:
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