"""
@ File:    tac-kbp2015.py
@ Author:  Zimu Wang
# Update:  May 3, 2022
@ Purpose: Convert the TAC KBP 2015 dataset.
"""
import re

import jsonlines
import os
from xml.dom.minidom import parse

from pathlib import Path 


class Config:
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration for the project (current) folder.
        self.PROJECT_FOLDER = "../../../data/tac_kbp_eng_event_nugget_detect_coref_2014-2015"

        # The configurations for the training data.
        self.TRAIN_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, "data/2015/training")
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "source")
        self.TRAIN_HOPPER_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "event_hopper")
        self.TRAIN_NUGGET_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "event_nugget")

        # The configurations for the testing data.
        self.TEST_DATA_FOLDER = os.path.join(self.PROJECT_FOLDER, "data/2015/eval")
        self.TEST_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "source")
        self.TEST_HOPPER_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "event_hopper")
        self.TEST_NUGGET_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "event_nugget")

        # save path 
        self.save_path = Path(os.path.join(self.PROJECT_FOLDER, "kbp2015"))
        self.save_path.mkdir(exist_ok=True)



def read_xml(nugget_folder, source_folder):
    """
    Read the annotated files and construct the data.
    :param nugget_folder: The path for the event_nugget folder.
    :param source_folder: The path for the source folder.
    :return: documents:   The set of the constructed documents.
    """
    # Initialise the document list.
    documents = list()

    # List all the files under the event_nugget folder.
    nugget_files = os.listdir(nugget_folder)
    for nugget_file in nugget_files:
        if nugget_file.endswith("xml"):
            # Initialise the structure of a document.
            document = {
                'id': str(),
                'text': str(),
                'events': list(),
                'negative_triggers': list()
            }

            # Extract the data from the xml file.
            dom_tree = parse(os.path.join(config.TRAIN_NUGGET_FOLDER, nugget_file))
            # document['id'] <- The id of each document (filename).
            document['id'] = dom_tree.documentElement.getAttribute("doc_id")
            # Extract the events from the xml file.
            events = dom_tree.documentElement.getElementsByTagName("event_mention")
            for event in events:
                trigger = event.getElementsByTagName("trigger")[0]
                event_dict = {
                    'type': event.getAttribute("subtype"),
                    'mentions': [{
                        'id': event.getAttribute("id"),
                        'trigger_word': trigger.childNodes[0].data,
                        'position': [int(trigger.getAttribute("offset")),
                                     int(trigger.getAttribute("offset")) + int(trigger.getAttribute("length"))]}],
                    'argument': dict()
                }
                document['events'].append(event_dict)
            documents.append(document)

    return read_source(documents, source_folder)


def read_source(documents, source_folder):
    """
    Extract the source texts from the corresponding file.
    :param documents:     The structured documents list.
    :param source_folder: Path of the source folder.
    """
    for document in documents:
        # Extract the text of each document.
        with open(os.path.join(source_folder, str(document['id'] + ".txt")),
                  "r") as source:
            document['text'] = source.read()

        # Find the number of XML elements before each character.
        xml_char = list()
        for i in range(len(document['text'])):
            # Retrieve the top i characters.
            text = document['text'][:i]
            # Find the length of the text after deleting the
            # XML elements and line breaks before the current index.
            num_del = len(re.sub('\n', '', re.sub('<.*?>', '', text)))
            xml_char.append(num_del)

        # Delete the XML characters from the texts.
        document['text'] = re.sub('<.*?>', '', document['text'])
        document['text'] = re.sub('\n', '', document['text'])

        # Find the number of spaces before the text.
        num_space = 0
        for token in document['text']:
            num_space += 1
            if token != " ":
                num_space -= 1
                break
        document['text'] = document['text'].strip()

        # Subtract the number of XML elements and line breaks.
        for event in document['events']:
            for mention in event['mentions']:
                mention['position'][0] = xml_char[mention['position'][0]] - num_space
                mention['position'][1] = xml_char[mention['position'][1]] - num_space

    return documents


def to_jsonl(filename, documents):
    """
    Write the manipulated dataset into jsonl file.
    :param filename:  Name of the saved file.
    :param documents: The manipulated dataset.
    :return:
    """
    with jsonlines.open(filename, "w") as w:
        w.write_all(documents)


if __name__ == '__main__':
    config = Config()

    # Construct the training and testing documents.
    train_documents = read_xml(config.TRAIN_NUGGET_FOLDER, config.TRAIN_SOURCE_FOLDER)
    test_documents = read_xml(config.TEST_NUGGET_FOLDER, config.TEST_SOURCE_FOLDER)

    # Save the documents into jsonl files.
    to_jsonl(os.path.join(config.save_path, "train.jsonl"), train_documents)
    to_jsonl(os.path.join(config.save_path, "test.jsonl"), test_documents)
