"""
@ File:    tac-kbp2014.py
@ Author:  Zimu Wang
# Update:  May 31, 2022
@ Purpose: Convert the TAC KBP 2014 dataset in document level.
"""
import copy

import jsonlines
import os
import re

from nltk.tokenize.punkt import PunktSentenceTokenizer


class Config(object):
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration for the current folder.
        self.PROJECT_FOLDER = "../../../"

        # The configurations for the training data.
        self.TRAIN_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'data/tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                   '2015/data/2014/training')
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'source')
        self.TRAIN_TOKEN_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'token_offset')
        self.TRAIN_ANNOTATION_TBF = os.path.join(self.TRAIN_DATA_FOLDER, 'annotation/annotation.tbf')

        # The configurations for the evaluation data.
        self.EVAL_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'data/tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                  '2015/data/2014/eval')
        self.EVAL_SOURCE_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, 'source')
        self.EVAL_TOKEN_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, 'token_offset')
        self.EVAL_ANNOTATION_TBF = os.path.join(self.EVAL_DATA_FOLDER, 'annotation/annotation.tbf')

        # The configurations for the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'data/tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                  '2015/TAC-KBP2014')
        if not os.path.exists(self.SAVE_DATA_FOLDER):
            os.mkdir(self.SAVE_DATA_FOLDER)


def read_annotation(ann_file_tbf, source_folder, token_folder):
    """
    Read the annotation.tbf and construct the structure for each document.
    :param ann_file_tbf:  Path of the annotation.tbf file.
    :param source_folder: Path for the source files.
    :param token_folder:  Path for the token_offset files.
    :return: documents:   The set of the documents.
    """
    # Initialise the document list.
    documents = list()
    # Initialise the structure of the first document.
    document = {
        'id': str(),
        'sentence': str(),
        'events': list(),
        'negative_triggers': list()
    }

    # Extract the annotation.tbf information.
    with open(ann_file_tbf) as ann_file:
        for line in ann_file:
            # Set the id of the document.
            if line.startswith('#BeginOfDocument'):
                document['id'] = line.strip().split(' ')[-1]
            # Extract the events of the document.
            elif line.startswith('brat_conversion'):
                _, _, event_id, offsets, trigger, event_type, _, _ = line.strip().split('\t')
                event = {
                    'type': event_type,
                    'triggers': [{'id': event_id, 'trigger_word': trigger, 'position': offsets}],
                    'arguments': list()
                }   # Set the position using offsets temporarily, which will be replaced later.
                document['events'].append(event)
            # Initialise the structure for the next document.
            elif line.startswith('#EndOfDocument'):
                documents.append(document)
                document = {
                    'id': str(),
                    'sentence': str(),
                    'events': list(),
                    'negative_triggers': list()
                }

    return read_source(documents, source_folder, token_folder)


def read_source(documents, source_folder, token_folder):
    """
    Extract the source sentences and replace the tokens' character positions.
    :param documents:     The structured documents list.
    :param source_folder: Path of the source folder.
    :param token_folder:  Path of the token_offset folder.
    :return: documents:   The manipulated documents list.
    """
    for document in documents:
        # Extract the sentence of each document.
        with open(os.path.join(source_folder, str(document['id'] + '.tkn.txt')),
                  'r') as source:
            document['sentence'] = source.read()

            # Find the number of xml characters before each character.
            xml_char = list()
            for i in range(len(document['sentence'])):
                # Retrieve the top i characters.
                sentence = document['sentence'][:i]
                # Find the length of the sentence after deleting the
                # xml elements and line breaks before the current index.
                # Delete the urls from the sentence.
                sentence_del = re.sub('<a href=(.*?)>http(.*?)</a>', ' ', sentence)
                # Delete the <DATETIME> elements from the sentence.
                sentence_del = re.sub('<DATETIME>(.*?)< / DATETIME>', ' ', sentence_del)
                # Delete the xml characters from the sentence.
                sentence_del = re.sub('<.*?>', ' ', sentence_del)
                # Delete the unpaired '< / DOC' element.
                sentence_del = re.sub('< / DOC', ' ', sentence_del)
                # Replace the line breaks using spaces.
                sentence_del = re.sub('\n', ' ', sentence_del)
                # Delete extra spaces.
                sentence_del = re.sub(' +', ' ', sentence_del)
                # Delete the spaces before the sentence.
                xml_char.append(len(sentence_del.lstrip()))

        # Replace the character position of each event.
        for event in document['events']:
            for trigger in event['triggers']:
                # Case 1: The event only covers one token.
                if len(trigger['position'].split(',')) == 1:
                    with open(os.path.join(token_folder, str(document['id'] + '.txt.tab'))) as offset:
                        for line in offset:
                            token_id, _, start_line, end_line = line.split('\t')
                            if token_id == trigger['position']:
                                trigger['position'] = [xml_char[int(start_line)],
                                                       xml_char[int(start_line)] + len(trigger['trigger_word'])]
                # Case 2: The event covers multiple tokens.
                else:
                    # Obtain the start and end token of the trigger.
                    positions = trigger['position'].split(',')
                    start_token, end_token = positions[0], positions[-1]
                    # Replace the token positions to character positions.
                    with open(os.path.join(token_folder, str(document['id'] + '.txt.tab'))) as offset:
                        start_pos, end_pos = 0, 0
                        for line in offset:
                            token_id, _, start_line, end_line = line.split('\t')
                            if token_id == start_token:
                                start_pos = int(start_line)
                            elif token_id == end_token:
                                end_pos = int(end_line.strip('\n'))
                        # Slice the trigger word for multiple spans.
                        trigger['trigger_word'] = document['sentence'][start_pos:end_pos + 1]
                        # Delete the line break within the trigger.
                        trigger['trigger_word'] = re.sub('\n', ' ', trigger['trigger_word'])
                        trigger['trigger_word'] = re.sub(' +', ' ', trigger['trigger_word'])
                        # Obtain the start and end position of the trigger.
                        trigger['position'] = [xml_char[start_pos], xml_char[start_pos] + len(trigger['trigger_word'])]

        # Delete the urls from the sentence.
        document['sentence'] = re.sub('<a href=(.*?)>http(.*?)</a>', ' ', document['sentence'])
        # Delete the <DATETIME> elements from the sentence.
        document['sentence'] = re.sub('<DATETIME>(.*?)< / DATETIME>', ' ', document['sentence'])
        # Delete the xml characters from the sentence.
        document['sentence'] = re.sub('<.*?>', ' ', document['sentence'])
        # Delete the unpaired '< / DOC' element.
        document['sentence'] = re.sub('< / DOC', ' ', document['sentence'])
        # Replace the line breaks using spaces.
        document['sentence'] = re.sub('\n', ' ', document['sentence'])
        # Delete extra spaces.
        document['sentence'] = re.sub(' +', ' ', document['sentence'])
        # Delete the spaces before the sentence.
        document['sentence'] = document['sentence'].strip()

    # Fix annotation errors & Delete wrong annotations.
    for document in documents:
        for event in document['events']:
            for trigger in event['triggers']:
                if document['sentence'][trigger['position'][0]:trigger['position'][1]] != \
                        trigger['trigger_word']:
                    # Manually fix an annotation error.
                    if document['id'] == 'ac6d66326a43c5c3fe546d82f66c4f16.cmp' and \
                            trigger['trigger_word'] == 'War':
                        trigger['position'][0], trigger['position'][1] = 2486, 2489
                    # Delete wrong annotations.
                    else:
                        event['triggers'].remove(trigger)

    assert check_position(documents)
    return sentence_tokenize(documents)


def sentence_tokenize(documents):
    """
    Tokenize the document into multiple sentences.
    :param documents:         The structured documents list.
    :return: documents_split: The split sentences' document.
    """
    # Initialise a list of the splitted documents.
    documents_split = list()
    documents_without_event = list()

    for document in documents:
        # Initialise the structure for the sentence without event.
        document_without_event = {
            'id': document['id'],
            'sentences': list()
        }

        # Tokenize the sentence of the document.
        sentence_pos = list()
        sentence_tokenize = list()
        for start_pos, end_pos in PunktSentenceTokenizer().span_tokenize(document['sentence']):
            sentence_pos.append([start_pos, end_pos])
            sentence_tokenize.append(document['sentence'][start_pos:end_pos])

        # Filter the events for each document.
        for i in range(len(sentence_tokenize)):
            # Initialise the structure of each sentence.
            sentence = {
                'id': document['id'] + '-' + str(i),
                'sentence': sentence_tokenize[i],
                'events': list(),
                'negative_triggers': list()
            }
            # Filter the events belong to the sentence.
            for event in document['events']:
                event_sent = {
                    'type': event['type'],
                    'triggers': list(),
                    'arguments': list()
                }
                for trigger in event['triggers']:
                    if sentence_pos[i][0] <= trigger['position'][0] < sentence_pos[i][1]:
                        event_sent['triggers'].append(copy.deepcopy(trigger))
                # Modify the start and end positions.
                if not len(event_sent['triggers']) == 0:
                    for triggers in event_sent['triggers']:
                        if not sentence['sentence'][triggers['position'][0]:triggers['position'][1]] == \
                                triggers['trigger_word']:
                            triggers['position'][0] -= sentence_pos[i][0]
                            triggers['position'][1] -= sentence_pos[i][0]
                    sentence['events'].append(event_sent)

            # Append the manipulated sentence into documents.
            if not len(sentence['events']) == 0:
                documents_split.append(sentence)
            else:
                document_without_event['sentences'].append(sentence['sentence'])

        # Append the sentence without event into the list.
        if len(document_without_event['sentences']) != 0:
            documents_without_event.append(document_without_event)

    # TODO: Search for methods to deal with wrong tokenisation.
    for document in documents_split:
        for event in document['events']:
            for trigger in event['triggers']:
                if document['sentence'][trigger['position'][0]:trigger['position'][1]] != \
                        trigger['trigger_word']:
                    event['triggers'].remove(trigger)

    assert check_position(documents_split)
    return documents_split, documents_without_event


def check_position(documents):
    """
    Check whether the position of each trigger is correct.
    :param documents: The set of the constructed documents.
    :return: True/False
    """
    for document in documents:
        for event in document['events']:
            for trigger in event['triggers']:
                if document['sentence'][trigger['position'][0]:trigger['position'][1]] != \
                        trigger['trigger_word']:
                    return False
    return True


def to_jsonl(filename, documents):
    """
    Write the manipulated dataset into jsonl file.
    :param filename:  Name of the saved file.
    :param documents: The manipulated dataset.
    :return:
    """
    with jsonlines.open(filename, 'w') as w:
        w.write_all(documents)


if __name__ == '__main__':
    config = Config()

    # Construct the training and evaluation documents.
    train_documents_sent, train_documents_without_event = \
        read_annotation(config.TRAIN_ANNOTATION_TBF, config.TRAIN_SOURCE_FOLDER, config.TRAIN_TOKEN_FOLDER)
    eval_documents_sent, eval_documents_without_event = \
        read_annotation(config.EVAL_ANNOTATION_TBF, config.EVAL_SOURCE_FOLDER, config.EVAL_TOKEN_FOLDER)

    # Save the documents into jsonl files.
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_train_sent.jsonl'), train_documents_sent)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_train_without_event.jsonl'),
             train_documents_without_event)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_eval_sent.jsonl'), eval_documents_sent)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_eval_without_event.jsonl'),
             eval_documents_without_event)
