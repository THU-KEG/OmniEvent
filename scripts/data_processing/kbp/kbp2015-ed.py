import copy
import re
import json
from typing import List, Dict, Union

import argparse
import jsonlines
import os

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from tqdm import tqdm
from xml.dom.minidom import parse


def read_xml(hopper_folder: str,
             source_folder: str):
    """Reads the annotation files and saves the annotation of event triggers.

    Reads the annotation files, extract the event trigger annotations, and saves them to a dictionary. Finally, the
    annotations of each document are stored in a list.

    Args:
        hopper_folder (`str`):
            A string representing the path of the folder containing the annotations of the documents.
        source_folder (`str`):
            A string indicating the path of the folder containing the source text of the documents.

    Returns:
        documents:
            A list of dictionaries containing each document's document id and the trigger annotations. The source text
            of each document is temporarily left blank, which will be extracted in the `read_source()` function. The
            processed `documents` is then sent to the `read_source()` function for source text extraction.
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


def read_source(documents: List[Dict[str, Union[str, List]]],
                source_folder: str):
    """Extracts the source text of each document, deletes the xml elements, and replaces the position annotations.

    Extracts the source text of each document replaces the position annotation of each trigger word to character-level
    annotation. The xml annotations (covered by "<>") and the urls (starting with "http") are then deleted from the
    source text, and then the position of each trigger is amended.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and event trigger annotations.
        source_folder (`str`):
            A string representing the path of the folder containing the source text of the documents.

    Returns:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and event trigger annotations of each
            document. The processed documents` is then sent to the `sentence_tokenize()` function for sentence
            tokenization.
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
            # Delete some special characters.
            text_del = re.sub("\x92", " ", text_del)
            text_del = re.sub("\x94", " ", text_del)
            text_del = re.sub("\x96", " ", text_del)
            text_del = re.sub("\x97", " ", text_del)
            # Replace the line breaks using spaces.
            text_del = re.sub("\n", " ", text_del)
            # Delete extra spaces.
            text_del = re.sub("\t", " ", text_del)
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
        # Delete some special characters.
        document["text"] = re.sub("\x92", " ", document["text"])
        document["text"] = re.sub("\x94", " ", document["text"])
        document["text"] = re.sub("\x96", " ", document["text"])
        document["text"] = re.sub("\x97", " ", document["text"])
        # Replace the line breaks using spaces.
        document["text"] = re.sub("\n", " ", document["text"])
        # Delete extra spaces.
        document["text"] = re.sub("\t", " ", document["text"])
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


def sentence_tokenize(documents: List[Dict[str, Union[str, List]]]):
    """Tokenizes the source text into sentences and matches the corresponding event triggers, arguments, and entities.

    Tokenizes the source text into sentences, and matches the event triggers, arguments, and entities that belong to
    each sentence. The sentences do not contain any triggers and entities are stored separately.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger annotations of each
            document.

    Returns:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers within.

        The processed `documents_split` and `documents_without_event` is then sent to the `add_spaces()` function
        for adding spaces beside punctuations.
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


def add_spaces(documents_split: List[Dict[str, Union[str, List]]],
               documents_without_event: List[Dict[str, Union[str, List[str]]]]):
    """Adds a space before and after the punctuations.

    Adds a space before and after punctuations, such as commas (","), full-stops ("?"), and question marks ("?"), of the
    source texts. However, the spaces should not be added besides the punctuations in a number, such as 1,000 and 1.5.
    Therefore, the spaces are added by the tokenization and detokenization to the source texts; after the process, the
    mention and position of the event trigger annotations are also amended.

    Args:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger annotations of each
            sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers within.

    Returns:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger annotations of each sentence within
            each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers within.
    """
    for document in tqdm(documents_split, desc="Adding spaces..."):
        punc_char = list()
        for i in range(len(document["text"])):
            # Retrieve the top i characters.
            text = document["text"][:i]
            # Tokenize and detokenize the retrieved string.
            punc_char.append(len(" ".join(word_tokenize(text))))
        punc_char.append(punc_char[-1])

        # Tokenize and detokenize the source text.
        document["text"] = " ".join(word_tokenize(document["text"]))

        # Fix the position of mentions due to an extra space before.
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

    # Tokenize and detokenize the sentences without events.
    for document in documents_without_event:
        for i in range(len(document["sentences"])):
            document["sentences"][i] = " ".join(word_tokenize(document["sentences"][i]))

    assert check_position(documents_split)
    return split_subwords(documents_split, documents_without_event)


def split_subwords(documents_split: List[Dict[str, Union[str, List]]],
                   documents_without_event: List[Dict[str, Union[str, List[str]]]]):
    """Splits the subwords into two separate words.

    Splits the subwords into two separate words for better PLM encodings. The example is as follows:

        - Original: Greece ’ s second-largest city
        - Processed: Greece ’ s second - largest city

    After splitting the subwords, the mention and position of the event trigger annotations are also amended.

    Args:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger annotations of each
            sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers within.

    Returns:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger annotations of each sentence within
            each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers within. The processed
            `documents_split` and `documents_without_event` are returned as final results.
    """
    for document in tqdm(documents_split, desc="Splitting subwords..."):
        punc_char = list()
        for i in range(len(document["text"])):
            # Retrieve the top i characters.
            text = document["text"][:i]
            # Split the subwords within the retrieved string.
            text = re.sub("-", " - ", text)
            punc_char.append(len(re.sub(" +", " ", text)))
        punc_char.append(punc_char[-1])

        # Tokenize and detokenize the source text.
        document["text"] = re.sub("-", " - ", document["text"])
        document["text"] = re.sub(" +", " ", document["text"])

        # Fix the position of mentions due to an extra space before.
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

    # Split the subwords within the sentences without events.
    for document in documents_without_event:
        for i in range(len(document["sentences"])):
            document["sentences"][i] = re.sub("-", " - ", document["sentences"][i])
            document["sentences"][i] = re.sub(" +", " ", document["sentences"][i])

    assert check_position(documents_split)
    return documents_split, documents_without_event


def check_position(documents: List[Dict[str, Union[str, List]]]) -> bool:
    """Checks whether the start and end positions correspond to the mention.

    Checks whether the string sliced from the source text based on the start and end positions corresponds to the
    mention.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger annotations of each document/
            sentence.

    Returns:
        Returns `False` if an inconsistency is found between the positions and the mention; otherwise, returns `True`.
    """
    for document in documents:
        for event in document["events"]:
            assert event["triggers"] != 0
            for trigger in event["triggers"]:
                if document["text"][trigger["position"][0]:trigger["position"][1]] \
                        != trigger["trigger_word"]:
                    return False
    return True


def to_jsonl(filename: str,
             save_dir: str,
             documents: List[Dict[str, Union[str, List]]]) -> None:
    """Writes the manipulated dataset into a jsonl file.

    Writes the manipulated dataset into a jsonl file; each line of the jsonl file corresponds to a piece of data.

    Args:
        filename (`str`):
            A string indicating the filename of the saved jsonl file.
        save_dir (`str`):
            A string indicating the directory to place the jsonl file.
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries indicating the `document_split` or the `document_without_event` dataset.
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


def token_pos_to_char_pos(tokens: List[str],
                          token_pos: List[int]) -> List[int]:
    """Converts the token-level position of a mention into character-level.

    Converts the token-level position of a mention into character-level by counting the number of characters before the
    start position of the mention. The end position could then be derived by adding the character-level start position
    and the length of the mention's span.

    Args:
        tokens (`List[str]`):
            A list of strings representing the tokens within the source text.
        token_pos (`List[int]`):
            A list of integers indicating the word-level start and end position of the mention.

    Returns:
        `List[int]`:
            A list of integers representing the character-level start and end position of the mention.
    """
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


def generate_negative_trigger(data: Dict[str, Union[str, List]],
                              none_event_instances: List[Dict[str, Union[str, List[str]]]]):
    """Generates negative triggers from the none-event instances.

    Generates negative triggers from the none-event instances, in which the tokens of the none-event sentences are
    regarded as negative triggers.

    Args:
        data (`Dict`):
            A dictionary containing the annotations of a sentence, including its id, source text, and the event trigger,
            argument, and entity annotations of the sentence.
        none_event_instances (`Dict`):
            A list of dictionaries containing the sentences that do not contain any event triggers and entities.

    Returns:
        A dictionary similar to the input dictionary but added the negative triggers annotations.
    """
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
    arg_parser = argparse.ArgumentParser(description="TAC-KBP2015")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/original/"
                                                            "tac_kbp_eng_event_nugget_detect_coref_2014-2015")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/TAC-KBP2015")
    args = arg_parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Construct the training and evaluation documents.
    train_documents_sent, train_documents_without_event \
        = read_xml(os.path.join(args.data_dir, "data/2015/training/event_hopper"),
                   os.path.join(args.data_dir, "data/2015/training/source"))
    eval_documents_sent, eval_documents_without_event \
        = read_xml(os.path.join(args.data_dir, "data/2015/eval/hopper"),
                   os.path.join(args.data_dir, "data/2015/eval/source"))

    # Save the documents into jsonl files.
    all_train_data = generate_negative_trigger(train_documents_sent, train_documents_without_event)
    json.dump(all_train_data, open(os.path.join(args.save_dir, 'train.json'), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, 'train.unified.jsonl'), args.save_dir, all_train_data)

    all_test_data = generate_negative_trigger(eval_documents_sent, eval_documents_without_event)
    json.dump(all_test_data, open(os.path.join(args.save_dir, 'test.json'), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, 'test.unified.jsonl'), args.save_dir, all_test_data)
