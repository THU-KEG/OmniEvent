"""
@ File:    tac-kbp2014.py
@ Author:  Zimu Wang
# Update:  May 17, 2022
@ Purpose: Convert the TAC KBP 2014 dataset.
"""
import jsonlines
import os
import re

from nltk.tokenize import sent_tokenize


class Config:
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration for the project (current) folder.
        self.PROJECT_FOLDER = "../../../data"

        # The configurations for the training data.
        self.TRAIN_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                   '2015/data/2014/training')
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'source')
        self.TRAIN_TOKEN_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'token_offset')
        self.TRAIN_ANNOTATION_TBF = os.path.join(self.TRAIN_DATA_FOLDER, 'annotation/annotation.tbf')

        # The configurations for the testing data.
        self.TEST_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                  '2015/data/2014/eval')
        self.TEST_SOURCE_FOLDER = os.path.join(self.TEST_DATA_FOLDER, 'source')
        self.TEST_TOKEN_FOLDER = os.path.join(self.TEST_DATA_FOLDER, 'token_offset')
        self.TEST_ANNOTATION_TBF = os.path.join(self.TEST_DATA_FOLDER, 'annotation/annotation.tbf')

        # The configurations for the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'tac_kbp_eng_event_nugget_detect_coref_2014-'
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
    # Initialise a list of the documents.
    documents = list()
    # Initialise the structure of a document.
    document = {
        'id': str(),
        'sentence': str(),
        'events': list(),
        'negative_triggers': list()
    }

    # Extract the annotation.tbf information.
    with open(ann_file_tbf) as ann_file:
        for line in ann_file:
            # Set the ID of the document.
            if line.startswith('#BeginOfDocument'):
                document['id'] = line.strip().split(' ')[-1]
            # Extract the events of the document.
            elif line.startswith('brat_conversion'):
                _, _, event_id, offsets, trigger, event_type, _, _ = line.strip().split('\t')
                event = {
                    'type': event_type,
                    'triggers': [{'id': event_id, 'trigger_word': trigger, 'position': offsets}],
                    'arguments': list()
                }  # Set the position using offsets temporarily, which will be replaced later.
                document['events'].append(event)
            # Initialise a structure at the end of a document.
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
    # Extract the sentences and replace the character positions.
    for document in documents:
        # Extract the sentence of each document.
        with open(os.path.join(source_folder, str(document['id'] + '.tkn.txt')),
                  'r') as source:
            document['sentence'] = source.read()

        # Find the number of characters in the clean sentence.
        xml_char = list()
        for i in range(len(document['sentence'])):
            # Retrieve the top i characters.
            sentence = document['sentence'][:i]
            # Find the length of the sentence after deleting the
            # XML elements and line breaks before the current index.
            sentence_del = re.sub('<DATETIME>(.*?)< / DATETIME>', ' ', sentence)
            sentence_del = re.sub('<.*?>', ' ', sentence_del)
            sentence_del = re.sub('< / DOC', ' ', sentence_del)
            sentence_del = re.sub('\n', ' ', sentence_del)
            sentence_del = re.sub(' +', ' ', sentence_del)
            num_del = len(sentence_del.lstrip())
            xml_char.append(num_del)

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
                        for line in offset:
                            token_id, _, start_line, end_line = line.split('\t')
                            if token_id == start_token:
                                start_pos = int(start_line)
                            elif token_id == end_token:
                                end_pos = int(end_line.strip('\n'))
                        # Slice the trigger word for multiple spans.
                        trigger['trigger_word'] = document['sentence'][start_pos:end_pos + 1]
                        trigger['position'] = [xml_char[start_pos],
                                               xml_char[start_pos] + len(trigger['trigger_word'])]

        # Delete the <DATETIME> elements from the sentences.
        document['sentence'] = re.sub('<DATETIME>(.*?)< / DATETIME>', ' ', document['sentence'])
        # Delete the XML characters from the sentences.
        document['sentence'] = re.sub('<.*?>', ' ', document['sentence'])
        # Delete the unpaired < / DOC characters.
        document['sentence'] = re.sub('< / DOC', ' ', document['sentence'])
        # Replace the line break using space.
        document['sentence'] = re.sub('\n', ' ', document['sentence'])
        # Delete extra spaces from the sentences.
        document['sentence'] = re.sub(' +', ' ', document['sentence'])
        # Delete the spaces before the sentence.
        document['sentence'] = document['sentence'].strip()

    return sentence_tokenize(documents)


def sentence_tokenize(documents):
    """
    Tokenize the document into multiple sentences.
    :param documents:         The structured documents list.
    :return: documents_split: The split sentences' document.
    """
    # Initialise a list of the split documents.
    documents_split = list()
    documents_without_event = list()

    for document in documents:
        # Initialise the structure for the sentence without event.
        document_without_event = {
            'id': document['id'],
            'sentences': list()
        }

        # Tokenize the sentence of the document.
        sentences_tokenize = sent_tokenize(document['sentence'])

        # Obtain the start and end position of each sentence.
        sentence_pos = list()
        for i in range(len(sentences_tokenize) - 1):
            start_pos = document['sentence'].find(sentences_tokenize[i])
            end_pos = document['sentence'].find(sentences_tokenize[i + 1])
            sentence_pos.append([start_pos, end_pos])
        # Set the start and end position of the last sentence.
        sentence_pos.append([document['sentence'].find(sentences_tokenize[-1]), len(document['sentence'])])

        # Filter the annotations for each document.
        for i in range(len(sentences_tokenize)):
            # Initialise the structure of each sentence.
            sentence = {
                'id': document['id'] + "-" + str(i),
                'sentence': sentences_tokenize[i],
                'events': list(),
                'negative_triggers': list()
            }

            # Append the events belong to the sentence.
            for event in document['events']:
                if sentence_pos[i][0] <= event['triggers'][0]['position'][0] < sentence_pos[i][1]:
                    sentence['events'].append(event)

            # Append the manipulated sentence into documents.
            if not len(sentence['events']) == 0:
                # Modify the start and end positions.
                for event in sentence['events']:
                    event['triggers'][0]['position'][0] -= sentence_pos[i][0]
                    event['triggers'][0]['position'][1] -= sentence_pos[i][0]
                documents_split.append(sentence)
            else:
                document_without_event['sentences'].append(sentence['sentence'])

        documents_without_event.append(document_without_event)

    return documents_split, documents_without_event


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

    # Construct the training and testing documents.
    train_documents, train_documents_without_event = \
        read_annotation(config.TRAIN_ANNOTATION_TBF, config.TRAIN_SOURCE_FOLDER, config.TRAIN_TOKEN_FOLDER)
    test_documents, test_documents_without_event = \
        read_annotation(config.TEST_ANNOTATION_TBF, config.TEST_SOURCE_FOLDER, config.TEST_TOKEN_FOLDER)

    # Save the documents into jsonl files.
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_training.jsonl'), train_documents)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_training_without_event.jsonl'),
             train_documents_without_event)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_eval.jsonl'), test_documents)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2014_eval_without_event.jsonl.jsonl'),
             test_documents_without_event)
