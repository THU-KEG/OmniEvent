import argparse
import json
import jsonlines
import os

from tqdm import tqdm
from typing import Dict, List, Tuple, Union


def read_data(data_folder: str) -> Tuple[List, List, List]:
    """Reads the WikiEvents datasets from the local path.

    Reads the WikiEvents datasets from the local path and returns the training, validation, and testing datasets.

    Args:
        data_folder (`str`):
            A string indicating the folder path of the WikiEvents datasets.

    Returns:
        train_data (`List[Dict]`), valid_data (`List[Dict]`), test_data (`List[Dict]`):
            Three lists of dictionaries representing the training, validation, and testing data.
    """
    train_data = list(jsonlines.open(os.path.join(data_folder, "train.jsonl")))
    valid_data = list(jsonlines.open(os.path.join(data_folder, "dev.jsonl")))
    test_data = list(jsonlines.open(os.path.join(data_folder, "test.jsonl")))

    return train_data, valid_data, test_data


def read_coref(data_folder: str) -> Tuple[List, List, List]:
    """Reads the coreferencial annotations of the WikiEvents datasets from the local path.

    Reads the coreferencial annotations of the WikiEvents datasets from the local path and returns the training,
    validation, and testing datasets.

    Args:
        data_folder (`str`):
            A string indicating the folder path of the coreferencial annotations of the WikiEvents datasets.

    Returns:
        train_data (`List[Dict]`), valid_data (`List[Dict]`), test_data (`List[Dict]`):
            Three lists of dictionaries representing the coreferencial annotations of the training, validation, and
            testing datasets.
    """
    train_coref = list(jsonlines.open(os.path.join(data_folder, "train.jsonlines")))
    valid_coref = list(jsonlines.open(os.path.join(data_folder, "dev.jsonlines")))
    test_coref = list(jsonlines.open(os.path.join(data_folder, "test.jsonlines")))

    return train_coref, valid_coref, test_coref


def convert_to_unified(data: List[Dict],
                       coref: List[Dict]) -> List[Dict]:
    """Converts the WikiEvents dataset to the unified OmniEvent format.

    Extracts the information from the original WikiEvents dataset, including the document ids, source texts, and event
    trigger, argument, and entity annotations of the documents, and converts them into the unified OmniEvent format.

    Args:
        data (`List[Dict]`):
            A list of dictionaries representing the original training/validation/testing dataset.
        coref (`List[Dict]`):
            A list of dictionaries representing the coreferential annotations of the dataset.

    Returns:
        documents (`List[Dict]`):
            A list of dictionaries representing the processed training/validation/testing dataset.
    """
    # Initialize a list for the processed dataset
    documents = list()

    # Convert the dataset into the unified OmniEvent format
    for one_data in tqdm(data, desc="Processing data..."):
        # Construct the basic structure of a document
        document = {
            "id": one_data["doc_id"],
            "text": " ".join(one_data["tokens"]),
            "events": list(),
            "negative_triggers": list(),
            "entities": list()
        }

        # Initialize a list storing the coreferential entities
        added_entities = list()
        # Extract the coreferential entity annotations from the dataset
        for one_coref in coref:
            if one_coref["doc_key"] == one_data["doc_id"]:
                # Extract the entity annotations for each cluster
                for cluster in one_coref["clusters"]:
                    # Initialize the structure for a cluster of entities
                    entity = {
                        "type": str(),
                        "mentions": list()
                    }
                    # Add the entity mentions to the structure
                    for entity_id in cluster:
                        for entity_mention in one_data["entity_mentions"]:
                            if entity_id == entity_mention["id"]:
                                # Extract the annotations of each entity mention
                                mention = {
                                    "id": entity_mention["id"],
                                    "mention": str(),
                                    "position": token_pos_to_char_pos(one_data["tokens"],
                                                                      [entity_mention["start"], entity_mention["end"]])
                                }
                                # Slice the entity mention based on its positions
                                mention["mention"] = document["text"][mention["position"][0]:mention["position"][1]]
                                # Update the entity type of the cluster
                                entity["type"] = entity_mention["entity_type"]
                                # Save the entity mention to the cluster
                                entity["mentions"].append(mention)
                                # Avoid adding the entity mention again
                                added_entities.append(entity_id)
                    document["entities"].append(entity)

        # Extract the singleton annotations from the dataset
        for entity_mention in one_data["entity_mentions"]:
            if not entity_mention["id"] in added_entities:
                # Extract the annotations of the entity mention
                entity = {
                    "type": entity_mention["entity_type"],
                    "mentions": [{
                        "id": entity_mention["id"],
                        "mention": str(),
                        "position": token_pos_to_char_pos(one_data["tokens"],
                                                          [entity_mention["start"], entity_mention["end"]])
                    }]
                }
                entity["mentions"][0]["mention"] = document["text"][entity["mentions"][0]["position"][0]:
                                                                    entity["mentions"][0]["position"][1]]
                document["entities"].append(entity)

        # Extract the event annotations from the dataset
        for event_mention in one_data["event_mentions"]:
            # Initialize the structure for an event
            event = {
                "type": event_mention["event_type"],
                "triggers": [{
                    "id": event_mention["id"],
                    "trigger_word": str(),
                    "position": token_pos_to_char_pos(one_data["tokens"],
                                                      [event_mention["trigger"]["start"],
                                                       event_mention["trigger"]["end"]]),
                    "arguments": list()
                }]
            }
            # Slice the trigger mention based on its positions
            event["triggers"][0]["trigger_word"] \
                = document["text"][event["triggers"][0]["position"][0]:event["triggers"][0]["position"][1]]

            # Initialize a list storing the coreferential arguments
            added_arguments = list()
            for one_coref in coref:
                if one_coref["doc_key"] == one_data["doc_id"]:
                    for cluster in one_coref["clusters"]:
                        # List the roles of each cluster in the event
                        cluster_roles = list()
                        for cluster_mention in cluster:
                            for argument_mention in event_mention["arguments"]:
                                if cluster_mention == argument_mention["entity_id"]:
                                    cluster_roles.append(argument_mention["role"])
                        # Add the coreferential arguments to the event
                        # We have observed the coreferential phenomenon only exists in some of the arguments
                        # w/ two argument mentions, w/o the arguments over three mentions
                        if len(cluster_roles) >= 2 and len(set(cluster_roles)) == 1:
                            # Initialize the structure for an argument
                            argument = {
                                "role": str(),
                                "mentions": list()
                            }
                            for cluster_mention in cluster:
                                for argument_mention in event_mention["arguments"]:
                                    if cluster_mention == argument_mention["entity_id"]:
                                        # Initialize the structure for an argument mention
                                        candidate_argument = {
                                            "id": argument_mention["entity_id"],
                                            "mention": str(),
                                            "position": -1
                                        }
                                        # Obtain the mention and position of the argument
                                        for entity in document["entities"]:
                                            for entity_mention in entity["mentions"]:
                                                if entity_mention["id"] == candidate_argument["id"]:
                                                    candidate_argument["mention"] = entity_mention["mention"]
                                                    candidate_argument["position"] = entity_mention["position"]
                                        assert candidate_argument["mention"] != ""
                                        assert candidate_argument["position"] != -1

                                        # Update the argument role of the cluster
                                        argument["role"] = argument_mention["role"]
                                        # Save the argument mention to the cluster
                                        argument["mentions"].append(candidate_argument)
                                        # Avoid adding the argument mention again
                                        added_arguments.append(candidate_argument["id"])
                            event["triggers"][0]["arguments"].append(argument)

            # Add the singleton arguments of the event
            for argument_mention in event_mention["arguments"]:
                if argument_mention["entity_id"] not in added_arguments:
                    # Initialize the structure for an argument
                    argument = {
                        "role": argument_mention["role"],
                        "mentions": [{
                            "id": argument_mention["entity_id"],
                            "mention": str(),
                            "position": -1
                        }]
                    }
                    # Obtain the mention and position of the argument
                    for entity in document["entities"]:
                        for one_entity in entity["mentions"]:
                            if one_entity["id"] == argument["mentions"][0]["id"]:
                                argument["mentions"][0]["mention"] = one_entity["mention"]
                                argument["mentions"][0]["position"] = one_entity["position"]
                    assert argument["mentions"][0]["mention"] != ""
                    assert argument["mentions"][0]["position"] != -1
                    event["triggers"][0]["arguments"].append(argument)

            document["events"].append(event)
        documents.append(document)

    return documents


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


def generate_negative_trigger(data: List[Dict]):
    """Generates negative triggers from the none-event instances.

    Generates negative triggers from the none-event instances, in which the tokens of the none-event sentences are
    regarded as negative triggers.

    Args:
        data (`List[Dict]`):
            A list of dictionaries containing the annotations of each sentence.

    Returns:
        A list of dictionaries similar to the input dictionary but added the negative triggers annotations.
    """
    for item in tqdm(data, desc="Generating negative triggers..."):
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

    return data


def gen_label2id_and_role2id(input_data: List[Dict],
                             save_dir: str) -> None:
    """Generates the correspondence between labels and ids, and roles and ids.

    Generates the correspondence between labels and ids, and roles and ids. Each label/role corresponds to a unique id.

    Args:
        input_data (`Dict`):
            A list of dictionaries containing the annotations of every sentence, including its id, source text, and the
            event trigger, argument, and entity annotations of the sentences.

    Returns:
        label_dict (`Dict[str, int]`):
            A dictionary containing the correspondence between the labels and their unique ids.
        role_dict (`Dict[str, int]`):
            A dictionary containing the correspondence between the roles and their unique ids.
    """
    label2id = dict(NA=0)
    role2id = dict(NA=0)
    print("We got %d instances" % len(input_data))
    for instance in input_data:
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


def to_jsonl(filename: str,
             save_dir: str,
             documents: List[Dict]) -> None:
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
    with jsonlines.open(filename, "w") as w:
        w.write_all(documents)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="WikiEvents")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/original/WikiEvents")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/WikiEvents")
    args = arg_parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Read the WikiEvents datasets and coreference annotations from the local path
    train_data, valid_data, test_data = read_data(args.data_dir)
    train_coref, valid_coref, test_coref = read_coref(args.data_dir)
    # Convert the WikiEvents datasets to the unified OmniEvent format
    train_documents, valid_documents, test_documents \
        = convert_to_unified(train_data, train_coref), convert_to_unified(valid_data, valid_coref), \
          convert_to_unified(test_data, test_coref)

    # Generate label2id and role2id
    gen_label2id_and_role2id(train_documents + valid_documents + test_documents, args.save_dir)

    # Save the documents into jsonl files
    train_documents = generate_negative_trigger(train_documents)
    json.dump(train_documents, open(os.path.join(args.save_dir, "train.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "train.unified.jsonl"), args.save_dir, train_documents)

    valid_documents = generate_negative_trigger(valid_documents)
    json.dump(valid_documents, open(os.path.join(args.save_dir, "valid.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "valid.unified.jsonl"), args.save_dir, valid_documents)

    test_documents = generate_negative_trigger(test_documents)
    json.dump(test_documents, open(os.path.join(args.save_dir, "test.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "test.unified.jsonl"), args.save_dir, test_documents)
