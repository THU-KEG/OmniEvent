import copy
import re
import json
import jsonlines
import os

from nltk.tokenize.punkt import PunktSentenceTokenizer
from tqdm import tqdm
from xml.dom.minidom import parse

from utils import token_pos_to_char_pos, generate_negative_trigger


class Config(object):
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration of the current folder.
        self.DATA_FOLDER = "../../../data"

        # The configurations of the training data.
        self.TRAIN_DATA_FOLDER = os.path.join(self.DATA_FOLDER, "tac_kbp_eng_event_nugget_detect_coref_2014-2015/data"
                                                                "/2015/training")
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "source")
        self.TRAIN_HOPPER_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "event_hopper")
        self.TRAIN_NUGGET_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "event_nugget")

        # The configurations of the evaluation data.
        self.EVAL_DATA_FOLDER = os.path.join(self.DATA_FOLDER, "tac_kbp_eng_event_nugget_detect_coref_2014-2015/data"
                                                               "/2015/eval")
        self.EVAL_SOURCE_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, "source")
        self.EVAL_HOPPER_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, "hopper")
        self.EVAL_NUGGET_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, "nugget")

        # The configuration of the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.DATA_FOLDER, "processed", "TAC-KBP2015")
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

    # List all the files under the event_hopper folder.
    hopper_files = os.listdir(hopper_folder)
    # Construct the document of each annotation data.
    for hopper_file in tqdm(hopper_files, desc="Reading hopper..."):
        # Initialise the structure of each document.
        document = {
            "id": str(),
            "text": str(),
            "events": list(),
            "negative_triggers": list(),
            "entities": list()
        }

        # Parse the data from the xml file.
        dom_tree = parse(os.path.join(hopper_folder, hopper_file))
        # Set the id (filename) as document id.
        document["id"] = dom_tree.documentElement.getAttribute("doc_id")

        # Extract the hoppers from the xml file.
        hoppers = dom_tree.documentElement.getElementsByTagName("hoppers")[0] \
                                          .getElementsByTagName("hopper")
        for hopper in hoppers:
            # Initialise a dictionary for each hopper.
            hopper_dict = {
                "type": hopper.getElementsByTagName("event_mention")[0]
                              .getAttribute("subtype"),
                "triggers": list()
            }
            # Extract the mentions within each hopper.
            mentions = hopper.getElementsByTagName("event_mention")
            # Extract the trigger from each mention.
            for mention in mentions:
                trigger = mention.getElementsByTagName("trigger")[0]
                assert len(mention.getElementsByTagName("trigger")) == 1
                trigger_dict = {
                    "id": mention.getAttribute("id"),
                    "trigger_word": trigger.childNodes[0].data,
                    "position": [int(trigger.getAttribute("offset")),
                                 int(trigger.getAttribute("offset")) + len(trigger.childNodes[0].data)],
                    "arguments": list()
                }
                hopper_dict["triggers"].append(trigger_dict)
            document["events"].append(hopper_dict)

        # Save the processed document into the document list.
        documents.append(document)

    return read_source(documents, source_folder)


def read_source(documents, source_folder):
    """
    Extract the source texts from the corresponding file.
    :param documents:     The documents lists with triggers.
    :param source_folder: Path of the source folder.
    :return: documents:   The set of the constructed documents.
    """
    for document in tqdm(documents, desc="Reading source..."):
        # Extract the sentence of each document.
        with open(os.path.join(source_folder, str(document["id"] + ".txt")),
                  "r") as source:
            document["text"] = source.read()

        # Find the number of xml characters before each character.
        xml_char = list()
        for i in range(len(document["text"])):
            # Retrieve the top i characters.
            text = document["text"][:i]
            # Delete the <DATETIME> elements from the text.
            text_del = re.sub("<DATETIME>(.*?)< / DATETIME>", " ", text)
            # Delete the xml characters from the text.
            text_del = re.sub("<.*?>", " ", text_del)
            # Delete the unpaired "</DOC" element.
            text_del = re.sub("</DOC", " ", text_del)
            # Delete the url elements from the text.
            text_del = re.sub("http(.*?) ", " ", text_del)
            text_del = re.sub("amp;", " ", text_del)
            # Replace the line breaks using spaces.
            text_del = re.sub("\t", " ", text_del)
            text_del = re.sub("\n", " ", text_del)
            # Delete extra spaces.
            text_del = re.sub(" +", " ", text_del)
            # Delete the spaces before the text.
            xml_char.append(len(text_del.lstrip()))

        # Delete the <DATETIME> elements from the text.
        document["text"] = re.sub("<DATETIME>(.*?)< / DATETIME>", " ", document["text"])
        # Delete the xml characters from the text.
        document["text"] = re.sub("<.*?>", " ", document["text"])
        # Delete the unpaired "</DOC" element.
        document["text"] = re.sub("</DOC", " ", document["text"])
        # Delete the url elements from the text.
        document["text"] = re.sub("http(.*?) ", " ", document["text"])
        document["text"] = re.sub("amp;", " ", document["text"])
        # Replace the line breaks using spaces.
        document["text"] = re.sub("\t", " ", document["text"])
        document["text"] = re.sub("\n", " ", document["text"])
        # Delete extra spaces.
        document["text"] = re.sub(" +", " ", document["text"])
        # Delete the spaces before the text.
        document["text"] = document["text"].strip()

        # Subtract the number of xml elements and line breaks.
        for event in document["events"]:
            for trigger in event["triggers"]:
                trigger["position"][0] = xml_char[trigger["position"][0]]
                trigger["position"][1] = xml_char[trigger["position"][1]]

        # Remove the triggers from the xml elements.
        for event in document["events"]:
            for trigger in event["triggers"]:
                if not document["text"][trigger["position"][0]:trigger["position"][1]] \
                       == trigger["trigger_word"]:
                    event["triggers"].remove(trigger)
                    continue

        # Delete the event if there's no event within.
        for event in document["events"]:
            if len(event["triggers"]) == 0:
                document["events"].remove(event)
                continue

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

    for document in tqdm(documents, desc="Tokenizing sentence..."):
        # Initialise the structure for the sentence without event.
        document_without_event = {
            "id": document["id"],
            "sentences": list()
        }

        # Tokenize the sentence of the document.
        sentence_pos = list()
        sentence_tokenize = list()
        for start_pos, end_pos in PunktSentenceTokenizer().span_tokenize(document["text"]):
            sentence_pos.append([start_pos, end_pos])
            sentence_tokenize.append(document["text"][start_pos:end_pos])

        # Filter the events for each document.
        for i in range(len(sentence_tokenize)):
            # Initialise the structure of each sentence.
            sentence = {
                "id": document["id"] + "-" + str(i),
                "text": sentence_tokenize[i],
                "events": list(),
                "negative_triggers": list(),
                "entities": list()
            }
            # Filter the events belong to the sentence.
            for event in document["events"]:
                event_sent = {
                    "type": event["type"],
                    "triggers": list()
                }
                for trigger in event["triggers"]:
                    if sentence_pos[i][0] <= trigger["position"][0] < sentence_pos[i][1]:
                        event_sent["triggers"].append(copy.deepcopy(trigger))
                # Modify the start and end positions.
                if not len(event_sent["triggers"]) == 0:
                    for triggers in event_sent["triggers"]:
                        triggers["position"][0] -= sentence_pos[i][0]
                        triggers["position"][1] -= sentence_pos[i][0]
                    sentence["events"].append(event_sent)

            # Append the manipulated sentence into documents.
            if not len(sentence["events"]) == 0:
                documents_split.append(sentence)
            else:
                document_without_event["sentences"].append(sentence["text"])

        # Append the sentence without event into the list.
        if len(document_without_event["sentences"]) != 0:
            documents_without_event.append(document_without_event)

    assert check_position(documents_split)
    return add_spaces(documents_split, documents_without_event)


def add_spaces(documents_split, documents_without_event):
    for document in tqdm(documents_split, desc="Adding spaces..."):
        punc_char = list()
        for i in range(len(document["text"])):
            # Retrieve the top i characters.
            text = document["text"][:i]
            text_space = re.sub(",", " , ", text)
            text_space = re.sub("\.", " . ", text_space)
            text_space = re.sub(":", " : ", text_space)
            text_space = re.sub(";", " : ", text_space)
            text_space = re.sub("\?", " ? ", text_space)
            text_space = re.sub("!", " ! ", text_space)
            text_space = re.sub("'", " ' ", text_space)
            text_space = re.sub("\"", " \" ", text_space)
            text_space = re.sub("\(", " ( ", text_space)
            text_space = re.sub("\)", " ) ", text_space)
            text_space = re.sub("\[", " [ ", text_space)
            text_space = re.sub("\]", " ] ", text_space)
            text_space = re.sub("\{", " { ", text_space)
            text_space = re.sub("\}", " } ", text_space)
            text_space = re.sub("-", " - ", text_space)
            text_space = re.sub("/", " / ", text_space)
            text_space = re.sub("_", " _ ", text_space)
            text_space = re.sub("\*", " * ", text_space)
            text_space = re.sub("`", " ` ", text_space)
            text_space = re.sub("‘", " ‘ ", text_space)
            text_space = re.sub("’", " ’ ", text_space)
            text_space = re.sub("“", " “ ", text_space)
            text_space = re.sub("”", " ” ", text_space)
            text_space = re.sub("…", " … ", text_space)
            text_space = re.sub(" +", " ", text_space)
            punc_char.append(len(text_space.lstrip()))
        punc_char.append(punc_char[-1])

        document["text"] = re.sub(",", " , ", document["text"])
        document["text"] = re.sub("\.", " . ", document["text"])
        document["text"] = re.sub(":", " : ", document["text"])
        document["text"] = re.sub(";", " ; ", document["text"])
        document["text"] = re.sub("\?", " ? ", document["text"])
        document["text"] = re.sub("!", " ! ", document["text"])
        document["text"] = re.sub("'", " ' ", document["text"])
        document["text"] = re.sub("\"", " \" ", document["text"])
        document["text"] = re.sub("\(", " ( ", document["text"])
        document["text"] = re.sub("\)", " ) ", document["text"])
        document["text"] = re.sub("\[", " [ ", document["text"])
        document["text"] = re.sub("\]", " ] ", document["text"])
        document["text"] = re.sub("\{", " { ", document["text"])
        document["text"] = re.sub("\}", " } ", document["text"])
        document["text"] = re.sub("-", " - ", document["text"])
        document["text"] = re.sub("/", " / ", document["text"])
        document["text"] = re.sub("_", " _ ", document["text"])
        document["text"] = re.sub("\*", " * ", document["text"])
        document["text"] = re.sub("`", " ` ", document["text"])
        document["text"] = re.sub("‘", " ‘ ", document["text"])
        document["text"] = re.sub("’", " ’ ", document["text"])
        document["text"] = re.sub("“", " “ ", document["text"])
        document["text"] = re.sub("”", " ” ", document["text"])
        document["text"] = re.sub("…", " … ", document["text"])
        document["text"] = re.sub(" +", " ", document["text"]).strip()

        for event in document["events"]:
            for trigger in event["triggers"]:
                trigger["position"][0] = punc_char[trigger["position"][0]]
                trigger["position"][1] = punc_char[trigger["position"][1]]
                trigger["trigger_word"] = document["text"][trigger["position"][0]:trigger["position"][1]]
                if trigger["trigger_word"].startswith(" "):
                    trigger["position"][0] += 1
                    trigger["trigger_word"] = document["text"][trigger["position"][0]:trigger["position"][1]]
                if trigger["trigger_word"].endswith(" "):
                    trigger["position"][1] -= 1
                    trigger["trigger_word"] = document["text"][trigger["position"][0]:trigger["position"][1]]

    for document in documents_without_event:
        for i in range(len(document["sentences"])):
            document["sentences"][i] = re.sub(",", " , ", document["sentences"][i])
            document["sentences"][i] = re.sub("\.", " . ", document["sentences"][i])
            document["sentences"][i] = re.sub(":", " : ", document["sentences"][i])
            document["sentences"][i] = re.sub(";", " : ", document["sentences"][i])
            document["sentences"][i] = re.sub("\?", " ? ", document["sentences"][i])
            document["sentences"][i] = re.sub("!", " ! ", document["sentences"][i])
            document["sentences"][i] = re.sub("'", " ' ", document["sentences"][i])
            document["sentences"][i] = re.sub("\"", " \" ", document["sentences"][i])
            document["sentences"][i] = re.sub("\(", " ( ", document["sentences"][i])
            document["sentences"][i] = re.sub("\)", " ) ", document["sentences"][i])
            document["sentences"][i] = re.sub("\[", " [ ", document["sentences"][i])
            document["sentences"][i] = re.sub("\]", " ] ", document["sentences"][i])
            document["sentences"][i] = re.sub("\{", " { ", document["sentences"][i])
            document["sentences"][i] = re.sub("\}", " } ", document["sentences"][i])
            document["sentences"][i] = re.sub("-", " - ", document["sentences"][i])
            document["sentences"][i] = re.sub("/", " / ", document["sentences"][i])
            document["sentences"][i] = re.sub("_", " _ ", document["sentences"][i])
            document["sentences"][i] = re.sub("\*", " * ", document["sentences"][i])
            document["sentences"][i] = re.sub("`", " ` ", document["sentences"][i])
            document["sentences"][i] = re.sub("‘", " ‘ ", document["sentences"][i])
            document["sentences"][i] = re.sub("’", " ’ ", document["sentences"][i])
            document["sentences"][i] = re.sub("“", " “ ", document["sentences"][i])
            document["sentences"][i] = re.sub("”", " ” ", document["sentences"][i])
            document["sentences"][i] = re.sub("…", " … ", document["sentences"][i])
            document["sentences"][i] = re.sub(" +", " ", document["sentences"][i]).strip()

    assert check_position(documents_split)
    return documents_split, documents_without_event


def check_position(documents):
    """
    Check whether the position of each trigger is correct.
    :param documents: The set of the constructed documents.
    :return: True/False
    """
    for document in documents:
        for event in document["events"]:
            assert event["triggers"] != 0
            for trigger in event["triggers"]:
                if document["text"][trigger["position"][0]:trigger["position"][1]] \
                        != trigger["trigger_word"]:
                    return False
    return True


def to_jsonl(filename, save_dir, documents):
    """
    Write the manipulated dataset into jsonl file.
    :param filename:  Name of the saved file.
    :param documents: The manipulated dataset.
    :return:
    """
    label2id = dict(NA=0)
    role2id = dict(NA=0)
    print("We got %d instances" % len(documents))
    for instance in documents:
        for event in instance["events"]:
            event["type"] = ".".join(event["type"].split("_"))
            if event["type"] not in label2id:
                label2id[event["type"]] = len(label2id)
            for trigger in event["triggers"]:
                for argument in trigger["arguments"]:
                    if argument["role"] not in role2id:
                        role2id[argument["role"]] = len(role2id)
    if "train" in filename:
        json.dump(label2id, open(os.path.join(save_dir, "label2id.json"), "w"))
        json.dump(role2id, open(os.path.join(save_dir, "role2id.json"), "w"))
    with jsonlines.open(filename, 'w') as w:
        w.write_all(documents)


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
                    "trigger_word": tokens[i],
                    "position": token_pos_to_char_pos(tokens, [i, i+1])
                }
                refined_sen_events["negative_triggers"].append(_none_event)
            none_event_data.append(refined_sen_events)
    all_data = data + none_event_data
    return all_data


if __name__ == "__main__":
    config = Config()

    # Construct the training and evaluation documents.
    train_documents_sent, train_documents_without_event \
        = read_xml(config.TRAIN_HOPPER_FOLDER, config.TRAIN_SOURCE_FOLDER)
    eval_documents_sent, eval_documents_without_event \
        = read_xml(config.EVAL_HOPPER_FOLDER, config.EVAL_SOURCE_FOLDER)

    # Save the documents into jsonl files.
    all_train_data = generate_negative_trigger(train_documents_sent, train_documents_without_event)
    json.dump(all_train_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'train.json'), "w"), indent=4)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'train.unified.jsonl'), config.SAVE_DATA_FOLDER, all_train_data)

    all_test_data = generate_negative_trigger(eval_documents_sent, eval_documents_without_event)
    json.dump(all_test_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'test.json'), "w"), indent=4)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'test.unified.jsonl'), config.SAVE_DATA_FOLDER, all_test_data)
