import json
import jsonlines
import os
import re

from tqdm import tqdm
from utils import token_pos_to_char_pos, generate_negative_trigger


class Config(object):
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration of the data folder.
        self.DATA_FOLDER = '../../../data/tac_kbp_eng_event_arg_comp_train_eval_2014-2015/data/2014'

        # The configurations for the training data.
        self.PILOT_DATA_FOLDER = os.path.join(self.DATA_FOLDER, 'pilot')
        self.PILOT_ANNOTATION_FOLDER = os.path.join(self.PILOT_DATA_FOLDER, 'LDC_assessments')
        self.PILOT_SOURCE_FOLDER = os.path.join(self.PILOT_DATA_FOLDER, 'source_documents')

        # The configurations for the evaluation data.
        self.EVAL_DATA_FOLDER = os.path.join(self.DATA_FOLDER, 'eval')
        self.EVAL_ANNOTATION_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, 'LDC_assessments')
        self.EVAL_SOURCE_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, 'source_documents')

        # The configuration of the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.DATA_FOLDER, 'processed', 'TAC-KBP2014')
        if not os.path.exists(self.SAVE_DATA_FOLDER):
            os.mkdir(self.SAVE_DATA_FOLDER)


def read_annotation(annotation_folder, source_folder, mode):
    """
    Read the event mentions and arguments in sentence level.
    :param annotation_folder: The path of the annotation folder.
    :param source_folder: The path of the source folder.
    :return: documents:   The manipulated documents list.
    """
    # Initialize a list for the documents.
    documents = list()
    # List the documents within the folder.
    annotation_files = os.listdir(annotation_folder)

    # Iterate the documents and extract the arguments.
    for annotation_file in tqdm(annotation_files):
        with open(os.path.join(annotation_folder, annotation_file),
                  'r') as annotation:
            # Set the sub-folder based on the file name.
            if mode == 'pilot':
                if annotation_file.startswith('bolt'):
                    sub_folder = 'df'
                    filename = annotation_file.rstrip('.out.tab')
                else:
                    sub_folder = 'nw'
                    filename = annotation_file.rstrip('.out.tab')
            else:
                if annotation_file.startswith('NYT'):
                    sub_folder = 'nw'
                    filename = annotation_file[:-4]
                else:
                    sub_folder = 'mpdf'
                    filename = str(annotation_file[:-4] + '.mpdf')

            # Read the original source text from the corresponding file.
            with open(os.path.join(source_folder, sub_folder, str(filename + '.txt')),
                      'r') as source:
                source_raw = source.read()

            for line in annotation:
                # Obtain the useful information in each line.
                data_line = line.split('\t')
                doc_id, event_type, role, sentence, offset \
                    = data_line[0], data_line[2], data_line[3], data_line[6], data_line[7]
                # Check whether the span comprises multiple spans.
                if ',' in sentence:
                    spans = sentence.split(',')
                    sentence_start, sentence_end = int(spans[0].split('-')[0]), (int(spans[-1].split('-')[1]) + 2)
                else:
                    sentence_start, sentence_end = int(sentence.split('-')[0]), (int(sentence.split('-')[1]) + 2)
                offset_start, offset_end = int(offset.split('-')[0]), (int(offset.split('-')[1]) + 1)

                # Save the arguments whose span is in the sentence span.
                if sentence_start <= offset_start < offset_end <= sentence_end:
                    document = {
                        'id': doc_id,
                        'text': str(),
                        'events': list(),
                        'entities': list()
                    }
                    # Set the text and obtain the reflection.
                    document['text'], xml_char = read_source(source_raw[sentence_start:sentence_end])
                    pos_start, pos_end = offset_start - sentence_start, offset_end - sentence_start
                    # Obtain the original mention annotation from the text.
                    event_mention = source_raw[offset_start:offset_end]
                    # Delete the xml annotations and line breaks within the mention.
                    event_mention = re.sub('<.*?>', ' ', event_mention)
                    event_mention = re.sub('\n', ' ', event_mention)
                    event_mention = re.sub(' +', ' ', event_mention)
                    # Save the annotations into a dictionary.
                    event_dict = {
                        'type': event_type,
                        'triggers': [{'id': 'N/A', 'trigger_word': 'N/A', 'position': 'N/A',
                                      'arguments': [{'role': role, 'mentions': [{
                                          'mention': event_mention,
                                          'position': [xml_char[pos_start],
                                                       xml_char[pos_start] + len(event_mention)]}]}]}]
                    }

                    # Fix some annotation errors.
                    if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith(' ') \
                            or event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith(',') \
                            or event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith('\n'):
                        event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].rstrip(', \n')
                        event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][1] -= 1

                    document['events'].append(event_dict)
                    documents.append(document)

    return documents


def read_source(source_raw):
    """
    Read the source file corresponding to each document.
    :param source_raw: The original text from the file.
    :return: source_processed, xml_char
    """
    # Find the number of xml characters before each character.
    xml_char = list()
    for i in range(len(source_raw)):
        # Retrieve the top i characters.
        text = source_raw[:i]
        # Find the length of the text after deleting the
        # xml elements and line breaks before the current index.
        # Delete the <DATELINE> elements from the text.
        text_del = re.sub('<DATELINE>(.*?)</DATELINE>', ' ', text)
        # Delete the xml characters from the text.
        text_del = re.sub('<.*?>', ' ', text_del)
        # Delete the url elements from the text.
        text_del = re.sub('http(.*?) ', ' ', text_del)
        # Replace the line breaks using spaces.
        text_del = re.sub('\n', ' ', text_del)
        # Delete extra spaces.
        text_del = re.sub(' +', ' ', text_del)
        # Delete the spaces before the text.
        xml_char.append(len(text_del.lstrip()))

    # Delete the <DATETIME> elements from the text.
    source_processed = re.sub('<DATELINE>(.*?)</DATELINE>', ' ', source_raw)
    # Delete the xml characters from the text.
    source_processed = re.sub('<.*?>', ' ', source_processed)
    # Delete the url elements from the text.
    source_processed = re.sub('http(.*?) ', ' ', source_processed)
    # Replace the line breaks using spaces.
    source_processed = re.sub('\n', ' ', source_processed)
    # Delete extra spaces.
    source_processed = re.sub(' +', ' ', source_processed)
    # Delete the spaces before the text.
    source_processed = source_processed.strip()

    return source_processed, xml_char


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

    # Obtain the pilot and evaluation data.
    pilot_documents = read_annotation(config.PILOT_ANNOTATION_FOLDER, config.PILOT_SOURCE_FOLDER, 'pilot')
    eval_documents = read_annotation(config.EVAL_ANNOTATION_FOLDER, config.EVAL_SOURCE_FOLDER, 'eval')

    # # Save the documents into jsonl files.
    # all_train_data = generate_negative_trigger(train_documents_sent, train_documents_without_event)
    # json.dump(all_train_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'train.json'), "w"), indent=4)
    # to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'train.unified.jsonl'), all_train_data)
    #
    # all_test_data = generate_negative_trigger(eval_documents_sent, eval_documents_without_event)
    # json.dump(all_test_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'test.json'), "w"), indent=4)
    # to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'test.unified.jsonl'), all_test_data)
