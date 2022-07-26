import os 
import json
import random 
from pathlib import Path 

def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item)+"\n")
    

def merge(data_dir):
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
        for line in f.readlines():
            item = json.loads(line.strip())
            item["source"] = "<maven>"
            all_train.append(item)
    
    # dev
    all_dev = []
    ere = []
    with open(os.path.join(data_dir, "ace2005-dygie/dev.unified.jsonl")) as f:
        for line in f.readlines():
            ere.append(json.loads(line.strip()))
    for item in ere:
        item["source"] = "<ere>"
        all_dev.append(item)
    maven_dev_test = []
    with open(os.path.join(data_dir, "MAVEN/valid.unified.jsonl")) as f:
        for line in f.readlines():
            item = json.loads(line.strip())
            item["source"] = "<maven>"
            maven_dev_test.append(item)
    num_maven_dev = len(maven_dev_test)
    all_dev.extend(maven_dev_test[:num_maven_dev//2])
    
    # test 
    all_test = []
    ere = json.load(open(os.path.join(data_dir, "TAC-KBP2017/test.json")))
    with open(os.path.join(data_dir, "ace2005-dygie/test.unified.jsonl")) as f:
        for line in f.readlines():
            ere.append(json.loads(line.strip()))
    for item in ere:
        item["source"] = "<ere>"
        all_test.append(item)
    all_test.extend(maven_dev_test[num_maven_dev//2:])

    print("All train: %d, all dev: %d, all test: %d" % (len(all_train), len(all_dev), len(all_test)))
    save_dir = Path("../../../data/processed/all-ed")
    save_dir.mkdir(exist_ok=True)
    save_jsonl(all_train, os.path.join(save_dir, "train.unified.jsonl"))
    save_jsonl(all_dev, os.path.join(save_dir, "dev.unified.jsonl"))
    save_jsonl(all_test, os.path.join(save_dir, "test.unified.jsonl"))


if __name__ == "__main__":
    merge("../../../data/processed")





