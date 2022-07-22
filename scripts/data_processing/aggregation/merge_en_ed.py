import os 
import json




def merge(data_dir):
    all_train = []
    ere = json.load(open("ace2005/train.json"))
    ere += json.load(open("ace2005/test.json"))
    ere += json.load(open("ere/LDC2015E29.json"))
    ere += json.load(open("ere/LDC2015E68.json"))
    ere += json.load(open("ere/LDC2015E78.json"))
    ere += json.load(open("TAC-KBP2014/train.json"))
    ere += json.load(open("TAC-KBP2014/test.json"))
    ere += json.load(open("TAC-KBP2015/train.json"))
    ere += json.load(open("TAC-KBP2015/test.json"))
    ere += json.load(open("TAC-KBP2016/pilot.json"))
    ere += json.load(open("TAC-KBP2016/test.json"))
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
    ere = json.load(open("ace2005/valid.json"))
    for item in ere:
        item["source"] = "<ere>"
        all_dev.append(item)
    
    maven_dev_test = []
    with open(os.path.join(data_dir, "MAVEN/valid.unified.jsonl")) as f:
        for line in f.readlines():
            item = json.loads(line.strip())
            item["source"] = "<maven>"
            maven_dev_test.append(item)
    
    # test 
    all_test = []
    ere = json.load(open(os.path.join(data_dir, "TAC-KBP2017/test.json")))
    for item in ere:
        item["source"] = "<ere>"
