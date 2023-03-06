import copy
import json
from typing import List, Dict, Union

import argparse
import jsonlines
import os
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from tqdm import tqdm
from xml.dom.minidom import parse

from utils import token_pos_to_char_pos, generate_negative_trigger


def read_eval(eval_gold_folder_df: str,
              eval_gold_folder_nw: str,
              eval_source_folder_df: str,
              eval_source_folder_nw: str):
    """Reads the files and construct the evaluation dataset.

    Reads the files and construct the evaluation dataset, including the source text from the discussion forum text (df)
    and newswire (nw). Construct the df and nw datasets separately and combine them for return.

    Args:
        eval_gold_folder_df (`str`):
            A string representing the path of the folder containing the annotations of the df documents.
        eval_gold_folder_nw (`str`):
            A string representing the path of the folder containing the annotations of the nw documents.
        eval_source_folder_df (`str`):
            A string representing the path of the folder containing the source text of the df documents.
        eval_source_folder_nw (`str`):
            A string representing the path of the folder containing the source text of the nw documents.

    Returns:
        `List[Dict[str, Union[str, List]]]`
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each sentence within each document of the evaluation dataset.

        `List[Dict[str, Union[str, List[str]]]]`:
             A list of dictionaries containing the sentences not contain any triggers and entities within. of the
             evaluation dataset.
    """
    # Separately construct the df and nw documents.
    eval_documents_df_sent, eval_documents_df_without_event = \
        read_xml(eval_gold_folder_df, eval_source_folder_df, mode="eval")
    eval_documents_nw_sent, eval_documents_nw_without_event = \
        read_xml(eval_gold_folder_nw, eval_source_folder_nw, mode="eval")

    # Combine the two documents" lists together.
    return [*eval_documents_df_sent, *eval_documents_nw_sent],\
           [*eval_documents_df_without_event, *eval_documents_nw_without_event]


def read_xml(gold_folder: str,
             source_folder: str,
             mode: str):
    """Reads the annotation files and saves the annotation of event triggers, arguments, and entities.

    Reads the annotation files and extracts the event trigger, argument, and entity annotations and saves them to a
    dictionary. Finally, the annotations of each document are stored in a list.

    Args:
        gold_folder (`str`):
            A string representing the path of the folder containing the annotations of the documents.
        source_folder (`str`):
            A string representing the path of the folder containing the source text of the documents.
        mode (`str`):
            A string indicating the type of the dataset to construct, either "train" or "eval".

    Returns:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing each document's document id and the trigger, argument, and entity
            annotations. The source text of each document is temporarily left blank, which will be extracted in the
            `read_source()` method. The processed `documents` is then sent to the `read_source()` function for source
            text extraction.
    """
    # Initialise the document list.
    documents = list()

    # List all the files under the gold_standard folder.
    gold_files = os.listdir(gold_folder)
    # Construct the document of each annotation data.
    for gold_file in tqdm(gold_files, desc="Reading hoppers..."):
        # Initialise the structure of a document.
        document = {
            "id": str(),
            "text": str(),
            "events": list(),
            "negative_triggers": list(),
            "entities": list()
        }

        # Parse the data from the xml file.
        dom_tree = parse(os.path.join(gold_folder, gold_file))
        # Set the id (filename) as document id.
        document["id"] = dom_tree.documentElement.getAttribute("doc_id")

        # Extract the entities from the xml file.
        entities = dom_tree.documentElement.getElementsByTagName("entities")[0] \
                                           .getElementsByTagName("entity")
        for entity in entities:
            # Initialise a dictionary for each entity.
            entity_dict = {
                "type": entity.getAttribute("type"),
                "mentions": list(),
            }
            # Extract the mentions within each entity.
            mentions = entity.getElementsByTagName("entity_mention")
            for mention in mentions:
                # Delete the url elements from the mention.
                entity_mention = mention.getElementsByTagName("mention_text")[0].childNodes[0].data
                entity_mention = re.sub("<.*?>", "", entity_mention)
                entity_mention = re.sub("amp;", "", entity_mention)
                mention_dict = {
                    "id": mention.getAttribute("id"),
                    "mention": entity_mention,
                    "position": [int(mention.getAttribute("offset")),
                                 int(mention.getAttribute("offset")) + len(entity_mention)]
                }
                entity_dict["mentions"].append(mention_dict)
            document["entities"].append(entity_dict)

        # Extract the fillers from the xml file.
        fillers = dom_tree.documentElement.getElementsByTagName("fillers")[0] \
                                          .getElementsByTagName("filler")
        for filler in fillers:
            # Delete the url elements from the mention.
            filler_mention = filler.childNodes[0].data
            filler_mention = re.sub("<.*?>", "", filler_mention)
            filler_mention = re.sub("amp;", "", filler_mention)
            # Initialise a dictionary for each filler.
            filler_dict = {
                "type": filler.getAttribute("type"),
                "mentions": [{
                    "id": filler.getAttribute("id"),
                    "mention": filler_mention,
                    "position": [int(filler.getAttribute("offset")),
                                 int(filler.getAttribute("offset")) + len(filler_mention)]}]
                }
            document["entities"].append(filler_dict)

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
            # Extract the triggers from each mention.
            for mention in mentions:
                trigger = mention.getElementsByTagName("trigger")[0]
                assert len(mention.getElementsByTagName("trigger")) == 1
                trigger_word = re.sub("<.*?>", "", trigger.childNodes[0].data)
                trigger_word = re.sub("amp;", "", trigger_word)
                trigger_dict = {
                    "id": mention.getAttribute("id"),
                    "trigger_word": trigger_word,
                    "position": [int(trigger.getAttribute("offset")),
                                 int(trigger.getAttribute("offset")) + len(trigger_word)],
                    "arguments": list()
                }

                # Extract the arguments for each trigger.
                arguments = mention.getElementsByTagName("em_arg")
                for argument in arguments:
                    # Classify the type of the argument (entity/filler).
                    if argument.getAttribute("entity_id") == "":
                        arg_id = argument.getAttribute("filler_id")
                    else:
                        arg_id = argument.getAttribute("entity_id")
                    # Initialise a flag for whether the entity id exists.
                    flag = 0
                    # Justify whether the argument being added.
                    for added_argument in trigger_dict["arguments"]:
                        if argument.getAttribute("role") == added_argument["role"] \
                                and arg_id == added_argument["id"]:
                            # Classify the type of the argument and obtain its mention id.
                            if argument.getAttribute("entity_mention_id") == "":
                                mention_id = argument.getAttribute("filler_id")
                            else:
                                mention_id = argument.getAttribute("entity_mention_id")
                            argument_mention = argument.childNodes[0].data
                            argument_mention = re.sub("<.*?>", "", argument_mention)
                            argument_mention = re.sub("amp;", "", argument_mention)
                            mention_dict = {
                                "id": mention_id,
                                "mention": argument_mention,
                                "position": -1
                            }
                            # Match the position of the argument.
                            for entity in document["entities"]:
                                for entity_mention in entity["mentions"]:
                                    if entity_mention["id"] == mention_id:
                                        mention_dict["mention"] = entity_mention["mention"]
                                        mention_dict["position"] = entity_mention["position"]
                            assert mention_dict["position"] != -1
                            added_argument["mentions"].append(mention_dict)
                            flag = 1
                    # Initialise a new dictionary if the entity id not exists.
                    # The id of the argument will be deleted later.
                    if flag == 0:
                        # Classify the type of the argument and obtain its id.
                        if argument.getAttribute("entity_mention_id") == "":
                            mention_id = argument.getAttribute("filler_id")
                        else:
                            mention_id = argument.getAttribute("entity_mention_id")
                        argument_mention = argument.childNodes[0].data
                        argument_mention = re.sub("<.*?>", "", argument_mention)
                        argument_mention = re.sub("amp;", "", argument_mention)
                        argument_dict = {
                            "id": arg_id,
                            "role": argument.getAttribute("role"),
                            "mentions": [{"id": mention_id,
                                          "mention": argument_mention,
                                          "position": -1}]
                        }
                        # Match the position of the argument.
                        for entity in document["entities"]:
                            for entity_mention in entity["mentions"]:
                                if entity_mention["id"] == mention_id:
                                    argument_dict["mentions"][0]["mention"] = entity_mention["mention"]
                                    argument_dict["mentions"][0]["position"] = entity_mention["position"]
                        assert argument_dict["mentions"][0]["position"] != -1
                        trigger_dict["arguments"].append(argument_dict)

                # Delete the id of each argument.
                for argument in trigger_dict["arguments"]:
                    del argument["id"]
                hopper_dict["triggers"].append(trigger_dict)

            document["events"].append(hopper_dict)
        documents.append(document)

    assert check_argument(documents)
    return read_source(documents, source_folder, mode)


def read_source(documents: List[Dict[str, Union[str, List]]],
                source_folder: str,
                mode: str):
    """Extracts the source text of each document and removes the xml elements.

    Extracts the source text of each document and removes the xml elements (covered by "<>"), url elements (start with
    "http"), and linebreaks within the source text. The position of trigger words, arguments, and entities are also
    amended after removing the xml elements.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each document.
        source_folder (`str`):
            A string representing the path of the folder containing the source text of the documents.
        mode (`str`):
            A string indicating the type of the dataset to construct, either "train" or "eval".

    Returns:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger, argument, and entity
            annotations of each document. The processed `documents` is then sent to the `clean_documents()` to remove
            the arguments and entities that within the xml elements of the original source text.
    """
    for document in tqdm(documents, desc="Reading source..."):
        # Configure the different file paths for df"s and nw"s.
        if mode == "pilot":
            if document["id"].startswith("AFP") or document["id"].startswith("APW") \
                    or document["id"].startswith("ENG") or document["id"].startswith("NYT") \
                    or document["id"].startswith("XIN"):
                source_path = os.path.join(source_folder, "nw", (document["id"].rstrip("-kbp") + ".xml"))
            else:
                source_path = os.path.join(source_folder, "mpdf", (document["id"] + ".mpdf.xml"))
        else:
            if document["id"].startswith("AFP") or document["id"].startswith("APW") \
                    or document["id"].startswith("ENG") or document["id"].startswith("NYT") \
                    or document["id"].startswith("XIN"):
                source_path = os.path.join(source_folder, (document["id"].rstrip("-kbp") + ".xml"))
            else:
                source_path = os.path.join(source_folder, "mpdf", (document["id"] + ".mpdf.xml"))

        # Extract the text of each document.
        with open(source_path, "r") as source:
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
            # Delete extra spaces before the text.
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
        # Delete extra spaces within the text.
        document["text"] = re.sub("\t", " ", document["text"])
        document["text"] = re.sub(" +", " ", document["text"])
        # Delete the spaces before the text.
        document["text"] = document["text"].strip()

        # Subtract the number of xml elements and line breaks.
        for event in document["events"]:
            for trigger in event["triggers"]:
                trigger["position"][0] = xml_char[trigger["position"][0]]
                trigger["position"][1] = xml_char[trigger["position"][1]]
        for entity in document["entities"]:
            for entity_mention in entity["mentions"]:
                entity_mention["position"][0] = xml_char[entity_mention["position"][0]]
                entity_mention["position"][1] = xml_char[entity_mention["position"][1]]

        # Manually fix some errors in entity position.
        for entity in document["entities"]:
            for mention in entity["mentions"]:
                if not document["text"][mention["position"][0]:mention["position"][1]] \
                        == mention["mention"]:
                    if mention["mention"] == "iPhone 6 plus space grey" \
                            or mention["mention"] == "members of the sports & entertainment industry" \
                            or mention["mention"] == "potential gay & lesbian donors" \
                            or mention["mention"] == "Smith & Wesson" \
                            or mention["mention"] == "his Smith & Wesson pistol" \
                            or mention["mention"] == "an international trade lawyer for Mitchell Silberberg & Knupp" \
                            or mention["mention"] == "Mitchell Silberberg & Knupp" \
                            or mention["mention"] == "Hoffmeyer's Firearms & Sporting Goods" \
                            or mention["mention"] == "owner of Hoffmeyer's Firearms & Sporting Goods in Grass Valley" \
                            or mention["mention"] == "Open Society Foundations":
                        mention["position"][1] = mention["position"][0] + len(mention["mention"])
                    elif mention["mention"] == "owners of  \"standard-essential\" patents":
                        mention["mention"] = "owners of \"standard-essential\" patents"

    assert check_argument(documents)
    return clean_documents(documents)


def clean_documents(documents: List[Dict[str, Union[str, List]]]):
    """Removes the entities and arguments within the xml elements of the original source text.

    Removes the entities and arguments within the xml elements of the original source text, Considering the xml elements
    have been removed from the source text in the `read_source()` function by constructing a new dataset, in which the
    event trigger, argument, and entity annotations are not within the xml elements.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger, argument, and entity
            annotations of each document.

    Returns:
        documents_clean (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger, argument, and entity
            annotations of each document, without the arguments and entities within the xml elements of the original
            source text. The processed `documents_clean` is then sent to the `sentence_tokenize()` function for sentence
            tokenization.
    """
    # Initialise the structure for the cleaned documents.
    documents_clean = list()

    # Clean the documents with correct elements.
    for document in tqdm(documents, desc="Cleaning document..."):
        # Initialise the structure for the cleaned document.
        document_clean = {
            "id": document["id"],
            "text": document["text"],
            "events": list(),
            "negative_triggers": list(),
            "entities": list()
        }

        # Save the entities not in the xml elements.
        for entity in document["entities"]:
            entity_clean = {
                "type": entity["type"],
                "mentions": list()
            }
            for mention in entity["mentions"]:
                if document_clean["text"][mention["position"][0]:mention["position"][1]] \
                       == mention["mention"]:
                    entity_clean["mentions"].append(mention)
            if len(entity_clean["mentions"]) != 0:
                document_clean["entities"].append(entity_clean)

        # Save the events and the cleaned arguments.
        for event in document["events"]:
            event_clean = {
                "type": event["type"],
                "triggers": list()
            }
            for trigger in event["triggers"]:
                trigger_clean = {
                    "id": trigger["id"],
                    "trigger_word": trigger["trigger_word"],
                    "position": trigger["position"],
                    "arguments": list()
                }
                for argument in trigger["arguments"]:
                    argument_clean = {
                        "role": argument["role"],
                        "mentions": list()
                    }
                    for mention in argument["mentions"]:
                        if document_clean["text"][mention["position"][0]:mention["position"][1]] \
                                == mention["mention"]:
                            argument_clean["mentions"].append(mention)
                    if len(argument_clean["mentions"]) != 0:
                        trigger_clean["arguments"].append(argument_clean)
                event_clean["triggers"].append(trigger_clean)
            document_clean["events"].append(event_clean)
        documents_clean.append(document_clean)

    assert check_argument(documents_clean)
    assert check_position(documents_clean)
    return sentence_tokenize(documents_clean)


def sentence_tokenize(documents: List[Dict[str, Union[str, List]]]):
    """Tokenizes the source text into sentences and matches the corresponding event triggers, arguments, and entities.

    Tokenizes the source text into sentences, and matches the event triggers, arguments, and entities that belong to
    each sentence. The sentences do not contain any triggers and entities are stored separately.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger, argument, and entity
            annotations of each document.

    Returns:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers and entities within.

        The processed `documents_split` and `documents_without_event` is then sent to the `add_spaces()` function
        for adding spaces beside punctuations.
    """
    # Initialise a list of the splitted documents.
    documents_split, documents_without_event = list(), list()

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
                        trigger_sent = {
                            "id": trigger["id"],
                            "trigger_word": trigger["trigger_word"],
                            "position": copy.deepcopy(trigger["position"]),
                            "arguments": list()
                        }
                        for argument in trigger["arguments"]:
                            argument_sent = {
                                "role": argument["role"],
                                "mentions": list()
                            }
                            for mention in argument["mentions"]:
                                if sentence_pos[i][0] <= mention["position"][0] < sentence_pos[i][1]:
                                    argument_sent["mentions"].append(copy.deepcopy(mention))
                            if not len(argument_sent["mentions"]) == 0:
                                trigger_sent["arguments"].append(argument_sent)
                        event_sent["triggers"].append(trigger_sent)

                # Modify the start and end positions.
                if not len(event_sent["triggers"]) == 0:
                    for trigger in event_sent["triggers"]:
                        trigger["position"][0] -= sentence_pos[i][0]
                        trigger["position"][1] -= sentence_pos[i][0]
                        for argument in trigger["arguments"]:
                            for mention in argument["mentions"]:
                                mention["position"][0] -= sentence_pos[i][0]
                                mention["position"][1] -= sentence_pos[i][0]
                    sentence["events"].append(event_sent)

            # Filter the entities belong to the sentence.
            for entity in document["entities"]:
                entity_sent = {
                    "type": entity["type"],
                    "mentions": list()
                }
                for mention in entity["mentions"]:
                    if sentence_pos[i][0] <= mention["position"][0] < sentence_pos[i][1]:
                        entity_sent["mentions"].append(copy.deepcopy(mention))
                if not len(entity_sent["mentions"]) == 0:
                    for mention in entity_sent["mentions"]:
                        mention["position"][0] -= sentence_pos[i][0]
                        mention["position"][1] -= sentence_pos[i][0]
                    sentence["entities"].append(entity_sent)

            # Append the manipulated sentence into the list of documents.
            if not (len(sentence["events"]) == 0 and len(sentence["entities"]) == 0):
                documents_split.append(sentence)
            else:
                document_without_event["sentences"].append(sentence["text"])

        # Append the sentence without event into the list.
        if len(document_without_event["sentences"]) != 0:
            documents_without_event.append(document_without_event)

    assert check_argument(documents_split)
    assert check_position(documents_split)
    return add_spaces(documents_split, documents_without_event)


def add_spaces(documents_split: List[Dict[str, Union[str, List]]],
               documents_without_event: List[Dict[str, Union[str, List[str]]]]):
    """Adds a space before and after the punctuations.

    Adds a space before and after punctuations, such as commas (","), full-stops ("?"), and question marks ("?"), of the
    source texts. However, the spaces should not be added besides the punctuations in a number, such as 1,000 and 1.5.
    Therefore, the spaces are added by the tokenization and detokenization to the source texts; after the process, the
    mention and position of the event trigger, argument, and entity annotations are also amended.

    Args:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger, argument, and entity
            annotations of each sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers and entities within.

    Returns:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers and entities within.
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
                for argument in trigger["arguments"]:
                    for mention in argument["mentions"]:
                        mention["position"][0] = punc_char[mention["position"][0]]
                        mention["position"][1] = punc_char[mention["position"][1]]
                        mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                        if mention["mention"].startswith(" "):
                            mention["position"][0] += 1
                            mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                        if mention["mention"].endswith(" "):
                            mention["position"][1] -= 1
                            mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
        for entity in document["entities"]:
            for mention in entity["mentions"]:
                mention["position"][0] = punc_char[mention["position"][0]]
                mention["position"][1] = punc_char[mention["position"][1]]
                mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                if mention["mention"].startswith(" "):
                    mention["position"][0] += 1
                    mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                if mention["mention"].endswith(" "):
                    mention["position"][1] -= 1
                    mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]

    # Tokenize and detokenize the sentences without events.
    for document in documents_without_event:
        for i in range(len(document["sentences"])):
            document["sentences"][i] = " ".join(word_tokenize(document["sentences"][i]))

    assert check_argument(documents_split)
    assert check_position(documents_split)
    return split_subwords(documents_split, documents_without_event)


def split_subwords(documents_split: List[Dict[str, Union[str, List]]],
                   documents_without_event: List[Dict[str, Union[str, List[str]]]]):
    """Splits the subwords into two separate words.

    Splits the subwords into two separate words for better PLM encodings. The example is as follows:

        - Original: Greece ’ s second-largest city
        - Processed: Greece ’ s second - largest city

    After splitting the subwords, the mention and position of the event trigger, argument, and entity annotations are
    also amended.

    Args:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id, source text, and the event trigger, argument, and entity
            annotations of each sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers and entities within.

    Returns:
        documents_split (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each sentence within each document.
        documents_without_event (`List[Dict[str, Union[str, List[str]]]]`):
            A list of dictionaries containing the sentences not contain any triggers and entities within. The processed
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
                for argument in trigger["arguments"]:
                    for mention in argument["mentions"]:
                        mention["position"][0] = punc_char[mention["position"][0]]
                        mention["position"][1] = punc_char[mention["position"][1]]
                        mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                        if mention["mention"].startswith(" "):
                            mention["position"][0] += 1
                            mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                        if mention["mention"].endswith(" "):
                            mention["position"][1] -= 1
                            mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
        for entity in document["entities"]:
            for mention in entity["mentions"]:
                mention["position"][0] = punc_char[mention["position"][0]]
                mention["position"][1] = punc_char[mention["position"][1]]
                mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                if mention["mention"].startswith(" "):
                    mention["position"][0] += 1
                    mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                if mention["mention"].endswith(" "):
                    mention["position"][1] -= 1
                    mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]

    # Split the subwords within the sentences without events.
    for document in documents_without_event:
        for i in range(len(document["sentences"])):
            document["sentences"][i] = re.sub("-", " - ", document["sentences"][i])
            document["sentences"][i] = re.sub(" +", " ", document["sentences"][i])

    assert check_argument(documents_split)
    assert check_position(documents_split)
    return documents_split, documents_without_event


def fix_tokenize(sentence_tokenize: List[str],
                 sentence_pos: List[List[str]]):
    """Fixes the wrong sentence tokenizations that affect the mention extraction.

    Fixes the wrong sentence tokenizations caused by `nltk.tokenize.punkt.PunktSentenceTokenizer` due to points (".")
    existing at the end of abbreviations, which are regarded as the end of a sentence from the sentence tokenization
    algorithm. Fix some wrong tokenizations that split a trigger word, an argument mention, or an entity mention into
    two sentences.

    Args:
        sentence_tokenize (`List[str]`):
            A list of strings indicating the sentences tokenized by `nltk.tokenize.punkt.PunktSentenceTokenizer`.
        sentence_pos (`List[List[int]]`):
            A list of lists containing each sentence's start and end character positions, corresponding to the sentences
            in `sentence_tokenize`.

    Returns:
        sentence_tokenize (`List[str]`):
            A list of strings indicating the sentences after fixing the wrong tokenizations.
        sentence_pos (`List[List[int]]`):
            A list of lists containing each sentence's start and end character positions, corresponding to the sentences
            in `sentence_tokenize`.
    """
    # Set a list for the deleted indexes.
    del_index = list()

    # Fix the errors in tokenization.
    for i in range(len(sentence_tokenize) - 1, -1, -1):
        if sentence_tokenize[i].endswith("U.S.") or sentence_tokenize[i].endswith("U.N.") \
                or sentence_tokenize[i].endswith("B.C.") or sentence_tokenize[i].endswith("H.F.") \
                or sentence_tokenize[i].endswith("J.C.") or sentence_tokenize[i].endswith("a.m.") \
                or sentence_tokenize[i].endswith("Jan.") or sentence_tokenize[i].endswith("Feb.") \
                or sentence_tokenize[i].endswith("Aug.") or sentence_tokenize[i].endswith("Sept.") \
                or sentence_tokenize[i].endswith("Oct.") or sentence_tokenize[i].endswith("Nov.") \
                or sentence_tokenize[i].endswith("Dec.") or sentence_tokenize[i].endswith("So.") \
                or sentence_tokenize[i].endswith("St.") or sentence_tokenize[i].endswith("Co.") \
                or sentence_tokenize[i].endswith("Ft.") or sentence_tokenize[i].endswith("W.E.B.") \
                or sentence_tokenize[i].endswith("T.D.S."):
            if i != len(sentence_tokenize) - 1:
                sentence_tokenize[i] = sentence_tokenize[i] + " " + sentence_tokenize[i + 1]
                sentence_pos[i][1] = sentence_pos[i + 1][1]
                del_index.append(i + 1)

    # Store the undeleted elements into new lists.
    new_sentence_tokenize, new_sentence_pos = list(), list()
    assert len(sentence_tokenize) == len(sentence_pos)
    for i in range(len(sentence_tokenize)):
        if i not in del_index:
            new_sentence_tokenize.append(sentence_tokenize[i])
            new_sentence_pos.append(sentence_pos[i])

    assert len(new_sentence_tokenize) == len(new_sentence_pos)
    return new_sentence_tokenize, new_sentence_pos


def check_argument(documents: List[Dict[str, Union[str, List]]]) -> bool:
    """Checks whether the argument and entity mentions with the same id are consistent.

    Checks whether the argument and entity mentions with the same id are consistent, considering various operations are
    conducted in each function.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each document/sentence.

    Returns:
        Returns `False` if an inconsistency is found between the argument and entity mentions with the same id;
        otherwise returns `True`.
    """
    for document in documents:
        for event in document["events"]:
            for trigger in event["triggers"]:
                for argument in trigger["arguments"]:
                    for arg_mention in argument["mentions"]:
                        for entity in document["entities"]:
                            for ent_mention in entity["mentions"]:
                                if arg_mention["id"] == ent_mention["id"]:
                                    if not arg_mention == ent_mention:
                                        return False
    return True


def check_position(documents: List[Dict[str, Union[str, List]]]) -> bool:
    """Checks whether the start and end positions correspond to the mention.

    Checks whether the string sliced from the source text based on the start and end positions corresponds to the
    mention.

    Args:
        documents (`List[Dict[str, Union[str, List]]]`):
            A list of dictionaries containing the document id and the event trigger, argument, and entity annotations of
            each document/sentence.

    Returns:
        Returns `False` if an inconsistency is found between the positions and the mention; otherwise, returns `True`.
    """
    for document in documents:
        # Check the positions of the events.
        for event in document["events"]:
            for trigger in event["triggers"]:
                if document["text"][trigger["position"][0]:trigger["position"][1]] != \
                        trigger["trigger_word"]:
                    return False
                for argument in trigger["arguments"]:
                    for mention in argument["mentions"]:
                        if document["text"][mention["position"][0]:mention["position"][1]] != \
                                mention["mention"]:
                            return False
        # Check the positions of the entities.
        for entity in document["entities"]:
            for mention in entity["mentions"]:
                if document["text"][mention["position"][0]:mention["position"][1]] != \
                        mention["mention"]:
                    return False
    return True


def to_jsonl(filename: str,
             save_dir: str,
             documents: List[Dict[str, Union[str, List]]]):
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
    json.dump(label2id, open(os.path.join(save_dir, "label2id.json"), "w"))
    json.dump(role2id, open(os.path.join(save_dir, "role2id.json"), "w"))
    with jsonlines.open(filename, "w") as w:
        w.write_all(documents)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="TAC-KBP2017")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/original/"
                                                            "tac_kbp_event_arg_comp_train_eval_2016-2017")
    arg_parser.add_argument("--source_dir", type=str, default="../../../data/original/"
                                                              "tac_kbp_eval_src_2016-2017")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/TAC-KBP2017")
    args = arg_parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Construct the evaluation documents.
    eval_documents_sent, eval_documents_without_event \
        = read_eval(os.path.join(args.data_dir, "data/2017/eval/eng/df/ere"),
                    os.path.join(args.data_dir, "data/2017/eval/eng/nw/ere"),
                    os.path.join(args.source_dir, "data/2017/eng/df"),
                    os.path.join(args.source_dir, "data/2017/eng/nw"))

    # Save the documents into jsonl file.
    all_test_data = generate_negative_trigger(eval_documents_sent, eval_documents_without_event)
    json.dump(all_test_data, open(os.path.join(args.save_dir, "test.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "test.unified.jsonl"), args.save_dir, all_test_data)
