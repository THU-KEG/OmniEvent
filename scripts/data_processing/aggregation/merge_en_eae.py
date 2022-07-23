import os 
import json
import random 
from pathlib import Path 


def remove_sub_word_annotations(input_data):
    """
        Some of KBP trigger and entity annotations are not a complete word, but a sub-word.
        We remove these annotations in order to simplify the training and evaluation process.

        Examples:
            Text: 'No trinket-gathering.'
            Trigger: 'gathering'

            Text: 'Joe Zollars After you finish it, I suggest you get "Angels and Demons" next.'
            Argument: 'Angels and Demons'
    """
    random.seed(42)
    del_sentence, del_trigger, del_argument, del_event = 0, 0, 0, 0
    output_data = []
    for i, sent in enumerate(input_data):
        for sp in ['\x96', '\x97']:
            sent['text'] = sent['text'].replace(sp, ' ')    # manually replace special tokens
        clean_events = []
        for event in sent['events']:
            clean_triggers = []
            for trigger in event['triggers']:
                left_pos = trigger['position'][0]
                if left_pos != 0 and sent['text'][left_pos - 1] != ' ':
                    del_trigger += 1
                else:
                    if 'arguments' in trigger:
                        clean_arguments = []
                        for arg in trigger['arguments']:
                            clean_mentions = []
                            for mention in arg['mentions']:
                                left_pos = mention['position'][0]
                                right_pos = mention['position'][1]
                                mention['mention_id'] = '{}-{}-{}'.format(mention['mention'], left_pos, right_pos)
                                if left_pos != 0 and sent['text'][left_pos - 1] != ' ':
                                    del_argument += 1
                                else:
                                    clean_mentions.append(mention)
                            arg['mentions'] = clean_mentions

                            if arg['mentions']:
                                clean_arguments.append(arg)
                        trigger['arguments'] = clean_arguments

                    clean_triggers.append(trigger)
            event['triggers'] = clean_triggers
            if event['triggers']:
                clean_events.append(event)
            else:
                del_event += 1
        sent['events'] = clean_events
        output_data.append(sent)

    print('deleted {} sentences, {} events, {} triggers, {} arguments'.format(del_sentence, del_event, del_trigger,
                                                                              del_argument))
    return output_data

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
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2016/pilot.json")))
    ere += json.load(open(os.path.join(data_dir, "TAC-KBP2016/test.json")))
    for item in ere:
        item["source"] = "<ere>"
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
    
    # test 
    all_test = []
    ere = json.load(open(os.path.join(data_dir, "TAC-KBP2017/test.json")))
    with open(os.path.join(data_dir, "ace2005-dygie/test.unified.jsonl")) as f:
        for line in f.readlines():
            ere.append(json.loads(line.strip()))
    for item in ere:
        item["source"] = "<ere>"
        all_test.append(item)

    print("All train: %d, all dev: %d, all test: %d" % (len(all_train), len(all_dev), len(all_test)))
    all_train = remove_sub_word_annotations(all_train)
    all_dev = remove_sub_word_annotations(all_dev)
    all_test = remove_sub_word_annotations(all_test)

    save_dir = Path("../../../data/processed/all-eae")
    save_dir.mkdir(exist_ok=True)

    save_jsonl(all_train, os.path.join(save_dir, "train.unified.jsonl"))
    save_jsonl(all_dev, os.path.join(save_dir, "dev.unified.jsonl"))
    save_jsonl(all_test, os.path.join(save_dir, "test.unified.jsonl"))


if __name__ == "__main__":
    merge("../../../data/processed")





