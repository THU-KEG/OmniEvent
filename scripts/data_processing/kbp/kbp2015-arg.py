import jsonlines
import os
import re

from tqdm import tqdm


class Config(object):
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration of the data folder.
        self.DATA_FOLDER = '../../../data/tac_kbp_eng_event_arg_comp_train_eval_2014-2015/data'

        # The configurations for the training data.
        self.TRAIN_ANNOTATION_FOLDER = os.path.join(self.DATA_FOLDER, '2015/training/assessment')
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.DATA_FOLDER, '2014/eval/source_documents')

        # The configurations for the evaluation data.
        self.EVAL_ANNOTATION_FOLDER = os.path.join(self.DATA_FOLDER, '2015/eval/assessments/complete/arguments')
        self.EVAL_SOURCE_FOLDER = os.path.join(self.DATA_FOLDER, '2015/eval/source_corpus')

        # The configuration of the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.DATA_FOLDER, 'processed', 'TAC-KBP2015')
        # if not os.path.exists(self.SAVE_DATA_FOLDER):
        #     os.mkdir(self.SAVE_DATA_FOLDER)


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
    annotation_files = [file for file in os.listdir(annotation_folder) if file != '.DS_Store']

    # Iterate the documents and extract the arguments.
    for annotation_file in tqdm(annotation_files):
        with open(os.path.join(annotation_folder, annotation_file),
                  'r') as annotation:
            # Set the sub-folder based on the file name.
            if mode == 'train':
                if annotation_file.startswith('NYT') or annotation_file.startswith('AFP') \
                        or annotation_file.startswith('APW') or annotation_file.startswith('XIN'):
                    sub_folder = 'nw'
                    filename = annotation_file
                else:
                    sub_folder = 'mpdf'
                    filename = str(annotation_file + '.mpdf')
            else:
                if annotation_file.startswith('NYT'):
                    sub_folder = 'nw'
                    filename = annotation_file
                else:
                    sub_folder = 'mpdf'
                    filename = str(annotation_file + '.mpdf')

            # Read the original source text from the corresponding file.
            with open(os.path.join(source_folder, sub_folder, str(filename + '.txt')),
                      'r') as source:
                source_raw = source.read()

            # Initialize a variable for the document's count.
            for line in annotation:
                # Obtain the useful information in each line.
                data_line = line.split('\t')
                doc_id, event_type, role, sentence, offset \
                    = data_line[0], data_line[2], data_line[3], data_line[6], data_line[7]
                # Check whether the span comprises multiple spans.
                if ',' in sentence:
                    spans = sentence.split(',')
                    start_index, end_index = list(), list()
                    for span in spans:
                        start_index.append(int(span.split('-')[0]))
                        end_index.append(int(span.split('-')[1]))
                    sentence_start, sentence_end = min(start_index), (max(end_index) + 2)
                else:
                    sentence_start, sentence_end = int(sentence.split('-')[0]), (int(sentence.split('-')[1]) + 2)
                offset_start, offset_end = int(offset.split('-')[0]), (int(offset.split('-')[1]) + 1)

                # Save the arguments whose span is in the sentence span.
                if sentence_start <= offset_start < offset_end <= sentence_end and data_line[8] == 'NIL':
                    document = {
                        'id': doc_id,
                        'text': str(),
                        'events': list(),
                        'entities': list()
                    }
                    # Set the text and obtain the reflection.
                    document['text'], xml_char = read_source(source_raw[sentence_start:sentence_end])
                    pos_start, pos_end = offset_start - sentence_start, offset_end - sentence_start
                    if pos_end < len(xml_char):
                        # Add the event information of the document.
                        event_dict = {
                            'type': event_type,
                            'triggers': [{'id': 'N/A', 'trigger_word': 'N/A', 'position': 'N/A',
                                          'arguments': [{'role': role, 'mentions': [{
                                              'mention': source_raw[offset_start:offset_end],
                                              'position': [xml_char[pos_start], xml_char[pos_end]]}]}]}]
                        }
                        # Fix some annotation errors.
                        if '<' == source_raw[offset_start] \
                                and (not source_raw[offset_start:offset_end].startswith('<P>')) \
                                and (not source_raw[offset_start:offset_end].startswith('</po')):
                            # Match the length of the xml characters.
                            xml_ann = re.findall('<.*?>', source_raw[offset_start:offset_end])
                            # Update the mention and position of the annotation.
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                = re.sub('<.*?>', '',
                                         event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] \
                                += len(xml_ann[0])
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][1] \
                                += len(xml_ann[0])
                        if '<' in source_raw[offset_start:offset_end] or '\n' in source_raw[offset_start:offset_end] \
                                or 'http' in source_raw[offset_start:offset_end] \
                                and (not source_raw[offset_start:offset_end].startswith('<P>')) \
                                and (not source_raw[offset_start:offset_end].startswith('</po')):
                            # Delete the xml characters and line breaks;
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                = re.sub('<.*?>', ' ',
                                         event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                = re.sub('\n', ' ',
                                         event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                = re.sub(' +', ' ',
                                         event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            # if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].startswith('P> '):
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].lstrip('P> ')
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] += 3
                            # if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].startswith('> '):
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].lstrip('> ')
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] += 2
                            # if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].startswith('http') \
                            #         and len(event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].split()) != 1:
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] \
                            #         += len(re.findall('http(.*?) ',
                            #                event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])[0])
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = re.sub('http(.*?) ', '',
                            #                  event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            # if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].startswith(' '):
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'][1:]
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] += 1
                            # if 'http' in event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         and (not event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].startswith('http')):
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = re.sub('http(.*?) ', ' ', event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = re.sub('http(.*?)', '',
                            #                  event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = re.sub(' +', ' ', event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][1] \
                            #         = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] \
                            #         + len(event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                            # if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith('< / P'):
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                            #         = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].lstrip('< / P')
                            #     event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][1] \
                            #         = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][0] \
                            #         + len(event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'])
                        if event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith(' ') \
                                or event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith(',') \
                                or event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].endswith('\n'):
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                = event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].rstrip(', \n')
                            event_dict['triggers'][0]['arguments'][0]['mentions'][0]['position'][1] -= 1
                        if '<' not in event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                and '>' not in event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                and 'http' not in event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'] \
                                and event_dict['triggers'][0]['arguments'][0]['mentions'][0]['mention'].strip() != '':
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


def token_pos_to_char_pos(tokens, token_pos):
    word_span = " ".join(tokens[token_pos[0]:token_pos[1]])
    char_start, char_end = -1, -1
    curr_pos = 0
    for i, token in enumerate(tokens):
        if i == token_pos[0]:
            char_start = curr_pos
            break
        curr_pos += len(token) + 1
    assert char_start != -1
    char_end = char_start + len(word_span)
    sen = " ".join(tokens)
    assert sen[char_start:char_end] == word_span
    return [char_start, char_end]


def generate_negative_trigger(data, none_event_instances):
    for item in data:
        tokens = item["text"].split()
        trigger_position = {i: False for i in range(len(tokens))}
        for event in item["events"]:
            for trigger in event["triggers"]:
                start_pos = len(item["text"][:trigger["position"][0]].split())
                end_pos = start_pos + len(trigger["trigger_word"].split())
                for pos in range(start_pos, end_pos):
                    trigger_position[pos] = True
        item["negative_triggers"] = []
        for i, token in enumerate(tokens):
            if trigger_position[i] or token == "":
                continue
            _event = {
                    "id": len(item["negative_triggers"]),
                    "trigger_word": tokens[i],
                    "position": token_pos_to_char_pos(tokens, [i, i+1])
            }
            item["negative_triggers"].append(_event)
    none_event_data = []
    for ins_idx, item in enumerate(none_event_instances):
        for sentence in item["sentences"]:
            refined_sen_events = dict(id="%s-%d"%(item["id"], len(data)+ins_idx))
            refined_sen_events["text"] = sentence
            refined_sen_events["events"] = []
            refined_sen_events["negative_triggers"] = []
            refined_sen_events["entities"] = []
            tokens = sentence.split()
            for i, token in enumerate(tokens):
                _none_event = {
                    "id": len(refined_sen_events["negative_triggers"]),
                    'trigger_word': tokens[i],
                    'position': token_pos_to_char_pos(tokens, [i, i+1])
                }
                refined_sen_events["negative_triggers"].append(_none_event)
            none_event_data.append(refined_sen_events)
    all_data = data + none_event_data
    return all_data


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
    train_documents = read_annotation(config.TRAIN_ANNOTATION_FOLDER, config.TRAIN_SOURCE_FOLDER, 'train')
    eval_documents = read_annotation(config.EVAL_ANNOTATION_FOLDER, config.EVAL_SOURCE_FOLDER, 'eval')

    # # Save the documents into jsonl files.
    # all_train_data = generate_negative_trigger(train_documents_sent, train_documents_without_event)
    # json.dump(all_train_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'train.json'), "w"), indent=4)
    # to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'train.unified.jsonl'), all_train_data)
    #
    # all_test_data = generate_negative_trigger(eval_documents_sent, eval_documents_without_event)
    # json.dump(all_test_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'test.json'), "w"), indent=4)
    # to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'test.unified.jsonl'), all_test_data)
