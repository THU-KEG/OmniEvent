"""
@ File:    tac-kbp2015.py
@ Author:  Zimu Wang
# Update:  May 17, 2022
@ Purpose: Convert the TAC KBP 2015 dataset.
"""
import re
import jsonlines
import os

from nltk.tokenize import sent_tokenize
from xml.dom.minidom import parse


class Config:
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration for the current folder.
        self.PROJECT_FOLDER = "../../../"

        # The configurations for the training data.
        self.TRAIN_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'data/tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                   '2015/data/2015/training')
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'source')
        self.TRAIN_HOPPER_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'event_hopper')
        self.TRAIN_NUGGET_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, 'event_nugget')

        # The configurations for the testing data.
        self.TEST_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'data/tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                  '2015/data/2015/eval')
        self.TEST_SOURCE_FOLDER = os.path.join(self.TEST_DATA_FOLDER, 'source')
        self.TEST_HOPPER_FOLDER = os.path.join(self.TEST_DATA_FOLDER, 'hopper')
        self.TEST_NUGGET_FOLDER = os.path.join(self.TEST_DATA_FOLDER, 'nugget')

        # The configurations for the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, 'data/tac_kbp_eng_event_nugget_detect_coref_2014-'
                                                                  '2015/tackbp2015')
        if not os.path.exists(self.SAVE_DATA_FOLDER):
            os.mkdir(self.SAVE_DATA_FOLDER)


def read_xml(hopper_folder, source_folder):
    """
    Read the annotated files and construct the data.
    :param hopper_folder: The path for the event_hopper folder.
    :param source_folder: The path for the source folder.
    :return: documents:   The set of the constructed documents.
    """
    # Initialise the document list.
    documents = list()

    # List all the files under the event_nugget folder.
    hopper_files = os.listdir(hopper_folder)
    for hopper_file in hopper_files:
        if hopper_file.endswith('xml'):
            # Initialise the structure of a document.
            document = {
                'id': str(),
                'sentence': str(),
                'events': list(),
                'negative_triggers': list()
            }

            # Extract the data from the XML file.
            dom_tree = parse(os.path.join(hopper_folder, hopper_file))
            # document['id'] <- The id of each document (filename).
            document['id'] = dom_tree.documentElement.getAttribute('doc_id')

            # Extract the hoppers from the XML file.
            hoppers = dom_tree.documentElement.getElementsByTagName('hoppers')[0] \
                                              .getElementsByTagName('hopper')
            for hopper in hoppers:
                # Initialise a dictionary for each hopper.
                hopper_dict = {
                    'type': hopper.getElementsByTagName('event_mention')[0]
                                  .getAttribute('subtype'),
                    'triggers': list(),
                    'arguments': list()
                }
                # Extract the mentions within each hopper.
                mentions = hopper.getElementsByTagName('event_mention')
                for mention in mentions:
                    # Extract the triggers.
                    trigger = mention.getElementsByTagName('trigger')[0]
                    trigger_dict = {
                        'id': mention.getAttribute('id'),
                        'trigger_word': trigger.childNodes[0].data,
                        'position': [int(trigger.getAttribute('offset')),
                                     int(trigger.getAttribute('offset')) + int(trigger.getAttribute('length'))]
                    }
                    hopper_dict['triggers'].append(trigger_dict)
                document['events'].append(hopper_dict)

            documents.append(document)

    return read_source(documents, source_folder)


def read_source(documents, source_folder):
    """
    Extract the source sentences from the corresponding file.
    :param documents:     The structured documents list.
    :param source_folder: Path of the source folder.
    :return: documents:   The set of the constructed documents.
    """
    for document in documents:
        # Extract the sentence of each document.
        with open(os.path.join(source_folder, str(document['id'] + '.txt')),
                  'r') as source:
            document['sentence'] = source.read()

        # Find the number of XML elements before each character.
        xml_char = list()
        for i in range(len(document['sentence'])):
            # Retrieve the top i characters.
            sentence = document['sentence'][:i]
            # Find the length of the sentence after deleting the
            # XML elements and line breaks before the current index.
            sentence_del = re.sub('<DATETIME>(.*?)< / DATETIME>', ' ', sentence)
            sentence_del = re.sub('<.*?>', ' ', sentence_del)
            sentence_del = re.sub('</DOC', ' ', sentence_del)
            sentence_del = re.sub('\n', ' ', sentence_del)
            sentence_del = re.sub(' +', ' ', sentence_del)
            num_del = len(sentence_del.lstrip())
            xml_char.append(num_del)

        # Delete the <DATETIME> elements from the sentences.
        document['sentence'] = re.sub('<DATETIME>(.*?)< / DATETIME>', ' ', document['sentence'])
        # Delete the XML characters from the sentences.
        document['sentence'] = re.sub('<.*?>', ' ', document['sentence'])
        # Delete the unpaired '</DOC' characters.
        document['sentence'] = re.sub('</DOC', ' ', document['sentence'])
        # Replace the line break using space.
        document['sentence'] = re.sub('\n', ' ', document['sentence'])
        # Delete extra spaces.
        document['sentence'] = re.sub(' +', ' ', document['sentence'])
        # Delete the space before the sentences.
        document['sentence'] = document['sentence'].strip()

        # Subtract the number of XML elements and line breaks.
        for event in document['events']:
            for trigger in event['triggers']:
                trigger['position'][0] = xml_char[trigger['position'][0]]
                trigger['position'][1] = xml_char[trigger['position'][1]]

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
                'id': document['id'] + '-' + str(i),
                'sentence': sentences_tokenize[i],
                'events': list(),
                'negative_triggers': list()
            }

            # Append the events belong to the sentence.
            for event in document['events']:
                new_event = {
                    'type': event['type'],
                    'triggers': list(),
                    'arguments': list()
                }
                for trigger in event['triggers']:
                    if sentence_pos[i][0] <= trigger['position'][0] < sentence_pos[i][1]:
                        new_event['triggers'].append(trigger)
                if not len(new_event['triggers']) == 0:
                    # Modify the start and end positions.
                    for trigger in new_event['triggers']:
                        trigger['position'][0] -= sentence_pos[i][0]
                        trigger['position'][1] -= sentence_pos[i][0]
                    sentence['events'].append(new_event)

            # Append the manipulated sentence into documents.
            if not len(sentence['events']) == 0:
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
        read_xml(config.TRAIN_HOPPER_FOLDER, config.TRAIN_SOURCE_FOLDER)
    test_documents, test_documents_without_event = \
        read_xml(config.TEST_HOPPER_FOLDER, config.TEST_SOURCE_FOLDER)

    # Save the documents into jsonl files.
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2015_training.jsonl'), train_documents)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2015_training_without_event.jsonl'),
             train_documents_without_event)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2015_eval.jsonl'), test_documents)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'tac-kbp2015_eval_without_event.jsonl.jsonl'),
             test_documents_without_event)
