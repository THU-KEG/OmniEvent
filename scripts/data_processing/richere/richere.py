import jsonlines
import json
import os
import random
import uuid

from tqdm import tqdm
from transformers import BertTokenizerFast
from typing import Dict, List, Union


def check_unk_in_text(input_text: str,
                      input_tokenizer) -> List[str]:
    """Checks the "UNK" tokens after tokenization.

    Checks the "UNK" tokens after tokenization, indicating the original token is out-of-vocabulary. The error tokens
    are stored in a list and returned.

    Args:
        input_text (`str`):
            A string representing the source text that to be tokenized.
        input_tokenizer:
            The tokenizer proposed for the tokenization process.

    Returns:
        error_tokens (`List[str]`):
            A list of strings indicating the out-of-vocabulary tokens in the original source text.
    """
    words = input_text.split()
    inputs = input_tokenizer(words, is_split_into_words=True)
    error_tokens = []
    for idx in set(range(len(words))).difference(set(inputs.word_ids()[1:-1])):
        error_tokens.append(words[idx])
    return error_tokens


def gen_label2id_and_role2id(input_data: List[Dict]):
    """Generates the correspondence between labels and ids, and roles and ids.

    Generates the correspondence between labels and ids, and roles and ids. Each label/role corresponds to a unique id.

    Args:
        input_data (`Dict`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        label_dict (`Dict[str, int]`):
            A dictionary containing the correspondence between the labels and their unique ids.
        role_dict (`Dict[str, int]`):
            A dictionary containing the correspondence between the roles and their unique ids.
    """
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


def gen_train_valid_set(input_data: List[Dict]):
    """Splits the training and validation set.

    Splits the training and validation set with a ratio of 0.8 and 0.2 randomly. 80% of the original dataset is regarded
    as the training set, while the rest of the 20% are regarded as the validation set.

    Args:
        input_data (`List[Dict]`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        train_set (`List[Dict]`), valid_set (`List[Dict]`):
            Two lists of dictionaries containing the training and validation datasets.
    """
    random.seed(42)
    random.shuffle(input_data)

    train_set = input_data[0:int(len(input_data) * 0.8)]
    valid_set = input_data[int(len(input_data) * 0.8):]

    return train_set, valid_set


def detect_sub_word_annotations(input_data: List[Dict]) -> List[Dict[str, Union[int, str]]]:
    """Detects the sub-word annotations in the dataset.

    Some KBP trigger and entity annotations are not a complete word, but a sub-word. The method detects and returns the
    sub-word annotations for further check and fix.

    Args:
        input_data (`List[Dict]`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        error (`List[Dict[str, Union[int, str]]]`)
            A list of dictionaries containing the sub-word triggers, arguments, and source texts.
    """
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


def remove_sub_word_annotations(input_data: List[Dict]) -> List[Dict]:
    """Remove the sub-word annotations within the dataset.

    Some KBP trigger and entity annotations are not a complete word, but a sub-word. The method removes these
    annotations in order to simplify the training and evaluation process.

    Examples:
        Text: "No trinket-gathering."
        Trigger: "gathering"

        Text: "Joe Zollars After you finish it, I suggest you get "Angels and Demons" next."
        Argument: "Angels and Demons"

    Args:
        input_data (`List[Dict]`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        output_data (`List[Dict]`):
            A list of dictionaries similar to the input dictionary but removed the sub-word annotations.
    """
    random.seed(42)
    del_sentence, del_trigger, del_argument, del_event = 0, 0, 0, 0
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    output_data = []
    all_unk = []
    for i, sent in enumerate(tqdm(input_data, desc="Removing subword annotations")):
        unk_words = check_unk_in_text(sent['text'], tokenizer)
        all_unk.extend(unk_words)
        for unk in unk_words:
            sent['text'] = sent['text'].replace(unk, ' ')    # manually replace special tokens
        assert len(check_unk_in_text(sent['text'], tokenizer)) == 0
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

    print('detected {} unk words: [{}]'.format(len(set(all_unk)), set(all_unk)))
    print('deleted {} sentences, {} events, {} triggers, {} arguments'.format(del_sentence, del_event, del_trigger,
                                                                              del_argument))

    return output_data


def remove_duplicate_event(input_data: List[Dict]):
    """Removes the duplicate event triggers and entities from the dataset.

    Removes the duplicate event mentions and entities from the dataset. The unique event triggers and arguments are
    stored in new lists, and then the original list is then replaced.

    Args:
        input_data (`List[Dict]`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        input_data (`List[Dict]`):
            A list of dictionaries similar to the input dictionary but without duplicate event triggers and mentions.
    """
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
    output_dir = "../../../data/processed/RichERE/"
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

    ldc2015e29 = list(jsonlines.open(input_dir + 'LDC2015E29/LDC2015E29.unified.jsonl'))
    ldc2015e68 = list(jsonlines.open(input_dir + 'LDC2015E68/LDC2015E68.unified.jsonl'))
    ldc2015e78 = list(jsonlines.open(input_dir + 'LDC2015E78/LDC2015E78.unified.jsonl'))

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
