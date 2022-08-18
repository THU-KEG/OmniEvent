import jsonlines
import json
import os
import uuid
import random
from tqdm import tqdm


def gen_label2id_and_role2id(input_data):
    label_dict = dict(NA=0)
    role_dict = dict(NA=0)
    for d in tqdm(input_data, desc='Generating Label2ID and Role2ID'):
        for event in d['events']:
            label = event['type']
            if label not in label_dict:
                label_dict[label] = len(label_dict)
            for trigger in event['triggers']:
                arguments = trigger['arguments']
                for arg in arguments:
                    role = arg['role']
                    if role not in role_dict:
                        role_dict[role] = len(role_dict)

    return label_dict, role_dict


def gen_train_valid_set(input_data):
    random.seed(42)
    random.shuffle(input_data)

    train_set = input_data[0:int(len(input_data) * 0.8)]
    valid_set = input_data[int(len(input_data) * 0.8):]

    return train_set, valid_set


def detect_sub_word_annotations(input_data):
    error = []
    for i, d in enumerate(input_data):
        tmp = dict(idx=i, trigger=[], argument=[], text=d['text'])
        for event in d['events']:
            for trigger in event['triggers']:
                left_pos = trigger['position'][0]
                if left_pos != 0 and d['text'][left_pos - 1] != ' ':
                    tmp['trigger'].append(trigger['trigger_word'])
                if 'arguments' in trigger:
                    for arg in trigger['arguments']:
                        for mention in arg['mentions']:
                            left_pos = mention['position'][0]
                            if left_pos != 0 and d['text'][left_pos - 1] != ' ':
                                tmp['argument'].append(mention['mention'])
        if tmp['trigger'] or tmp['argument']:
            error.append(tmp)
    return error


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


def remove_duplicate_event(input_data):
    for d in input_data:
        clean_events = []
        for event in d['events']:
            if event not in clean_events:
                clean_events.append(event)
        d['events'] = clean_events
        for entity in d['entities']:
            for mention in entity['mentions']:
                left_pos = mention['position'][0]
                right_pos = mention['position'][1]
                mention['mention_id'] = '{}-{}-{}'.format(mention['mention'], left_pos, right_pos)
    return input_data


if __name__ == "__main__":
    """
        This script is to collect train, valid and test data for KBP experiment.
        You have to process the following dataset in advance:
            ldc2015e29, ldc2015e68, ldc2015e78, 
            kbp2014, kbp2015, kbp2016, kbp2017

        Our setting is:
            kbp2017 as test set, and the rest data as train and valid sets.
    """
    input_dir = "../../../data/processed/"
    output_dir = "../../../data/processed/TAC-KBP/"
    os.makedirs(output_dir, exist_ok=True)
    is_eae_14_15 = False  # kbp2014 and kbp2015 do not have EAE data currently.
    dump = True

    kbp2014 = list(jsonlines.open(input_dir + 'TAC-KBP2014/train.unified.jsonl')) + list(
        jsonlines.open(input_dir + '/TAC-KBP2014/test.unified.jsonl'))
    kbp2015 = list(jsonlines.open(input_dir + 'TAC-KBP2015/train.unified.jsonl')) + list(
        jsonlines.open(input_dir + '/TAC-KBP2015/test.unified.jsonl'))
    kbp2016 = list(jsonlines.open(input_dir + 'TAC-KBP2016/pilot.unified.jsonl')) + list(
        jsonlines.open(input_dir + '/TAC-KBP2016/test.unified.jsonl'))
    kbp2017 = list(jsonlines.open(input_dir + 'TAC-KBP2017/test.unified.jsonl'))

    ldc2015e29 = list(jsonlines.open(input_dir + 'ere/LDC2015E29.unified.jsonl'))
    ldc2015e68 = list(jsonlines.open(input_dir + 'ere/LDC2015E68.unified.jsonl'))
    ldc2015e78 = list(jsonlines.open(input_dir + 'ere/LDC2015E78.unified.jsonl'))

    if is_eae_14_15:
        train_and_valid = ldc2015e29 + ldc2015e68 + ldc2015e78 + kbp2016 + kbp2014 + kbp2015
    else:
        train_and_valid = ldc2015e29 + ldc2015e68 + ldc2015e78 + kbp2016

    test = kbp2017
    train, valid = gen_train_valid_set(train_and_valid)

    label2id, role2id = gen_label2id_and_role2id(train_and_valid + test)

    train = remove_duplicate_event(train)
    valid = remove_duplicate_event(valid)
    test = remove_duplicate_event(test)

    error_train = detect_sub_word_annotations(train)
    error_valid = detect_sub_word_annotations(valid)
    error_test = detect_sub_word_annotations(test)

    train = remove_sub_word_annotations(train)
    valid = remove_sub_word_annotations(valid)
    test = remove_sub_word_annotations(test)

    assert len(detect_sub_word_annotations(train)) == 0
    assert len(detect_sub_word_annotations(valid)) == 0
    assert len(detect_sub_word_annotations(test)) == 0

    if dump:
        os.makedirs(output_dir, exist_ok=True)
        with jsonlines.open(os.path.join(output_dir, 'train.unified.jsonl'), 'w') as f:
            for t in train:
                jsonlines.Writer.write(f, t)

        with jsonlines.open(os.path.join(output_dir, 'valid.unified.jsonl'), 'w') as f:
            for v in valid:
                jsonlines.Writer.write(f, v)

        with jsonlines.open(os.path.join(output_dir, 'test.unified.jsonl'), 'w') as f:
            for t in test:
                jsonlines.Writer.write(f, t)

        with open(os.path.join(output_dir, 'label2id.json'), 'w', encoding='utf-8') as f:
            json.dump(label2id, f, indent=4, ensure_ascii=False)

        with open(os.path.join(output_dir, 'role2id.json'), 'w', encoding='utf-8') as f:
            json.dump(role2id, f, indent=4, ensure_ascii=False)

    print('collected {} training instances'.format(len(train)))
    print('collected {} valid instances'.format(len(valid)))
    print('collected {} test instances'.format(len(test)))

    print('#event_type: {}'.format(len(label2id)))
    print('#argument_role: {}'.format(len(role2id)))
