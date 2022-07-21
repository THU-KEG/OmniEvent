import copy
import json
import jsonlines
import os
import re

from nltk.tokenize.punkt import PunktSentenceTokenizer
from tqdm import tqdm

from utils import token_pos_to_char_pos, generate_negative_trigger


class Config(object):
    """
    The configurations of this project.
    """
    def __init__(self):
        # The configuration for the current folder.
        self.DATA_FOLDER = "../../../data"

        # The configurations for the training data.
        self.TRAIN_DATA_FOLDER = os.path.join(self.DATA_FOLDER, "tac_kbp_eng_event_nugget_detect_coref_2014-2015/data"
                                                                "/2014/training")
        self.TRAIN_SOURCE_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "source")
        self.TRAIN_TOKEN_FOLDER = os.path.join(self.TRAIN_DATA_FOLDER, "token_offset")
        self.TRAIN_ANNOTATION_TBF = os.path.join(self.TRAIN_DATA_FOLDER, "annotation/annotation.tbf")

        # The configurations for the evaluation data.
        self.EVAL_DATA_FOLDER = os.path.join(self.DATA_FOLDER, "tac_kbp_eng_event_nugget_detect_coref_2014-2015/data"
                                                               "/2014/eval")
        self.EVAL_SOURCE_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, "source")
        self.EVAL_TOKEN_FOLDER = os.path.join(self.EVAL_DATA_FOLDER, "token_offset")
        self.EVAL_ANNOTATION_TBF = os.path.join(self.EVAL_DATA_FOLDER, "annotation/annotation.tbf")

        # The configuration for the saving path.
        self.SAVE_DATA_FOLDER = os.path.join(self.DATA_FOLDER, "processed", "TAC-KBP2014")
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
        "id": str(),
        "text": str(),
        "events": list(),
        "negative_triggers": list(),
        "entities": list()
    }

    # Extract the annotations from annotation.tbf.
    with open(ann_file_tbf) as ann_file:
        for line in tqdm(ann_file, desc="Reading annotation..."):
            # Set the id of the document.
            if line.startswith("#BeginOfDocument"):
                document["id"] = line.strip().split(" ")[-1]
            # Extract the events of the document.
            elif line.startswith("brat_conversion"):
                _, _, event_id, offsets, trigger, event_type, _, _ = line.strip().split("\t")
                event = {
                    "type": event_type,
                    "triggers": [{"id": event_id, "trigger_word": trigger,
                                  "position": offsets, "arguments": list()}]
                }   # Set the position using offsets temporarily, which will be replaced later.
                document["events"].append(event)
            # Initialise the structure for the next document.
            elif line.startswith("#EndOfDocument"):
                documents.append(document)
                document = {
                    "id": str(),
                    "text": str(),
                    "events": list(),
                    "negative_triggers": list(),
                    "entities": list()
                }
            else:
                print("The regulation of %s has not been set." % line)

    return read_source(documents, source_folder, token_folder)


def read_source(documents, source_folder, token_folder):
    """
    Extract the source texts and replace the tokens' character positions.
    :param documents:     The structured documents list.
    :param source_folder: Path of the source folder.
    :param token_folder:  Path of the token_offset folder.
    :return: documents:   The manipulated documents list.
    """
    for document in tqdm(documents, desc="Reading source..."):
        # Extract the text of each document.
        with open(os.path.join(source_folder, str(document["id"] + ".tkn.txt")),
                  "r") as source:
            document["text"] = source.read()

        # Find the number of xml characters before each character.
        xml_char = list()
        for i in range(len(document["text"])):
            # Retrieve the first i characters of the source text.
            text = document["text"][:i]
            # Delete the <DATETIME> elements from the text.
            text_del = re.sub("<DATETIME>(.*?)< / DATETIME>", " ", text)
            # Delete the xml characters from the text.
            text_del = re.sub("<.*?>", " ", text_del)
            # Delete the unpaired "< / DOC" element.
            text_del = re.sub("< / DOC", " ", text_del)
            # Delete the url elements from the text.
            text_del = re.sub("http(.*?) ", " ", text_del)
            # Replace the line breaks using spaces.
            text_del = re.sub("\n", " ", text_del)
            # Delete extra spaces within the text.
            text_del = re.sub(" +", " ", text_del)
            # Delete the spaces before the text.
            xml_char.append(len(text_del.lstrip()))

        # Replace the character position of each trigger.
        for event in document["events"]:
            for trigger in event["triggers"]:
                # Case 1: The event trigger only covers one token.
                if len(trigger["position"].split(",")) == 1:
                    with open(os.path.join(token_folder, str(document["id"] + ".txt.tab"))) as offset:
                        for line in offset:
                            token_id, _, token_begin, token_end = line.split("\t")
                            if token_id == trigger["position"]:
                                trigger["position"] = [xml_char[int(token_begin)],
                                                       xml_char[int(token_begin)] + len(trigger["trigger_word"])]
                                # Manually fix an annotation that covers xml elements.
                                if document["id"] == "bolt-eng-DF-170-181122-8808533" and token_id == "t2649":
                                    trigger["position"] \
                                        = [xml_char[13250], xml_char[13250] + len(trigger["trigger_word"])]
                        assert type(trigger["position"]) != str
                # Case 2: The event trigger covers multiple tokens.
                else:
                    # Obtain the start and end token of the trigger.
                    positions = trigger["position"].split(",")
                    start_token, end_token = positions[0], positions[-1]
                    # Replace the token positions to character positions.
                    with open(os.path.join(token_folder, str(document["id"] + ".txt.tab"))) as offset:
                        start_pos, end_pos = 0, 0
                        for line in offset:
                            token_id, _, token_begin, token_end = line.split("\t")
                            if token_id == start_token:
                                start_pos = int(token_begin)
                            elif token_id == end_token:
                                end_pos = int(token_end.strip("\n"))
                        assert start_pos != 0 and end_pos != 0
                        # Slice the trigger word for multiple spans.
                        trigger["trigger_word"] = document["text"][start_pos:end_pos + 1]
                        # Delete the line break within the trigger.
                        trigger["trigger_word"] = re.sub("\n", " ", trigger["trigger_word"])
                        trigger["trigger_word"] = re.sub(" +", " ", trigger["trigger_word"])
                        # Obtain the start and end position of the trigger.
                        trigger["position"] = [xml_char[start_pos], xml_char[start_pos] + len(trigger["trigger_word"])]

        # Delete the <DATETIME> elements from the text.
        document["text"] = re.sub("<DATETIME>(.*?)< / DATETIME>", " ", document["text"])
        # Delete the xml characters from the text.
        document["text"] = re.sub("<.*?>", " ", document["text"])
        # Delete the unpaired "</DOC" element.
        document["text"] = re.sub("< / DOC", " ", document["text"])
        # Delete the url elements from the text.
        document["text"] = re.sub("http(.*?) ", " ", document["text"])
        # Replace the line breaks using spaces.
        document["text"] = re.sub("\n", " ", document["text"])
        # Delete extra spaces.
        document["text"] = re.sub(" +", " ", document["text"])
        # Delete the spaces before the text.
        document["text"] = document["text"].strip()

        # Fix some annotation errors within the dataset.
        for event in document["events"]:
            for trigger in event["triggers"]:
                if document["text"][trigger["position"][0]:trigger["position"][1]] \
                        != trigger["trigger_word"]:
                    # Manually fix some annotation errors within the dataset.
                    if document["text"][trigger["position"][0]:trigger["position"][1]] == "ant":
                        trigger["trigger_word"] = "anti-war"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("anti-war")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "Ant":
                        trigger["trigger_word"] = "Anti-war"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("Anti-war")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "pro":
                        trigger["trigger_word"] = "pro-war"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("pro-war")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "counter-t":
                        trigger["trigger_word"] = "counter-terrorism"
                        trigger["position"] \
                            = [trigger["position"][0], trigger["position"][0] + len("counter-terrorism")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "Counter-demons":
                        trigger["trigger_word"] = "Counter-demonstrations"
                        trigger["position"] \
                            = [trigger["position"][0], trigger["position"][0] + len("Counter-demonstrations")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "re-elect":
                        trigger["trigger_word"] = "re-election"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("re-election")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "w":
                        trigger["trigger_word"] = "wedding"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("wedding")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "'Wa":
                        trigger["trigger_word"] = "War"
                        trigger["position"] = [trigger["position"][0] + 1, trigger["position"][1] + 1]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "co-foun":
                        trigger["trigger_word"] = "co-founded"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("co-founded")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "Non-ele":
                        trigger["trigger_word"] = "Non-elected"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("Non-elected")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "'Terminall":
                        trigger["trigger_word"] = "Terminally"
                        trigger["position"] = [trigger["position"][0] + 1, trigger["position"][1] + 1]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "'Battere":
                        trigger["trigger_word"] = "Battered"
                        trigger["position"] = [trigger["position"][0] + 1, trigger["position"][1] + 1]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "CO-FOUN":
                        trigger["trigger_word"] = "CO-FOUNDER"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("CO-FOUNDER")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "post-ele":
                        trigger["trigger_word"] = "post-election"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("post-election")]
                    elif document["text"][trigger["position"][0]:trigger["position"][1]] == "r":
                        trigger["trigger_word"] = "resignation"
                        trigger["position"] = [trigger["position"][0], trigger["position"][0] + len("resignation")]
                    # Delete the annotations that belong to the xml characters.
                    else:
                        event["triggers"].remove(trigger)
                        continue

        # Delete the event if there's no triggers within.
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
        sentence_tokenize, sentence_pos = fix_tokenize(sentence_tokenize, sentence_pos)

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
    return documents_split, documents_without_event


def fix_tokenize(sentence_tokenize, sentence_pos):
    """
    Fix the wrong tokenization within a sentence.
    :param sentence_pos:      List of starting and ending position of each sentence.
    :param sentence_tokenize: The tokenized sentences list.
    :return: The fixed sentence position and tokenization lists.
    """
    # Set a list for the deleted indexes.
    del_index = list()

    # Fix the errors in tokenization.
    for i in range(len(sentence_tokenize) - 1, -1, -1):
        if sentence_tokenize[i].endswith("U.S."):
            if i not in del_index:
                sentence_tokenize[i] = sentence_tokenize[i] + " " + sentence_tokenize[i + 1]
                sentence_pos[i][1] = sentence_pos[i + 1][1]
            else:
                sentence_tokenize[i - 1] = sentence_tokenize[i - 1] + " " + sentence_tokenize[i]
                sentence_pos[i - 1][1] = sentence_pos[i][1]
            del_index.append(i)

    # Store the undeleted elements into new lists.
    new_sentence_tokenize = list()
    new_sentence_pos = list()
    assert len(sentence_tokenize) == len(sentence_pos)
    for i in range(len(sentence_tokenize)):
        if i not in del_index:
            new_sentence_tokenize.append(sentence_tokenize[i])
            new_sentence_pos.append(sentence_pos[i])

    assert len(new_sentence_tokenize) == len(new_sentence_pos)
    return new_sentence_tokenize, new_sentence_pos


def check_position(documents):
    """
    Check whether the position of each trigger is correct.
    :param documents: The set of the constructed documents.
    :return: True/False
    """
    for document in documents:
        for event in document["events"]:
            assert len(document["events"]) != 0
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


if __name__ == "__main__":
    config = Config()

    # Construct the training and evaluation documents.
    train_documents_sent, train_documents_without_event \
        = read_annotation(config.TRAIN_ANNOTATION_TBF, config.TRAIN_SOURCE_FOLDER, config.TRAIN_TOKEN_FOLDER)
    eval_documents_sent, eval_documents_without_event \
        = read_annotation(config.EVAL_ANNOTATION_TBF, config.EVAL_SOURCE_FOLDER, config.EVAL_TOKEN_FOLDER)

    # Save the documents into jsonl files.
    all_train_data = generate_negative_trigger(train_documents_sent, train_documents_without_event)
    json.dump(all_train_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'train.json'), "w"), indent=4)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'train.unified.jsonl'), config.SAVE_DATA_FOLDER, all_train_data)

    all_test_data = generate_negative_trigger(eval_documents_sent, eval_documents_without_event)
    json.dump(all_test_data, open(os.path.join(config.SAVE_DATA_FOLDER, 'test.json'), "w"), indent=4)
    to_jsonl(os.path.join(config.SAVE_DATA_FOLDER, 'test.unified.jsonl'), config.SAVE_DATA_FOLDER, all_test_data)
