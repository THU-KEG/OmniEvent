import argparse
import json
import jsonlines
import os
import re

from tqdm import tqdm
from typing import Dict, List, Tuple, Union


def read_data(data_folder: str) -> Tuple[List, List, List]:
    """Reads the RAMS datasets from the local path.

    Reads the RAMS datasets from the local path and returns the training, validation, and testing datasets.

    Args:
        data_folder (`str`):
            A string indicating the folder path of the RAMS datasets.

    Returns:
        train_data (`List[Dict]`), valid_data (`List[Dict]`), test_data (`List[Dict]`):
            Three lists of dictionaries representing the training, validation, and testing datasets.
    """
    train_data = list(jsonlines.open(os.path.join(data_folder, "data/train.jsonlines")))
    valid_data = list(jsonlines.open(os.path.join(data_folder, "data/dev.jsonlines")))
    test_data = list(jsonlines.open(os.path.join(data_folder, "data/test.jsonlines")))

    return train_data, valid_data, test_data


def convert_to_unified(data: List[Dict]):
    """Converts the RAMS dataset to the unified OmniEvent format.

    Extracts the information from the original RAMS dataset, including the document ids, source texts, and event
    trigger, argument, and entity annotations of the documents, and converts them into the unified OmniEvent format.

    Args:
        data (`List[Dict]`):
            A list of dictionaries representing the original training/validation/testing dataset.

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
            "id": one_data["doc_key"],
            "text": " ".join([" ".join(sentence) for sentence in one_data["sentences"]]),
            "events": list(),
            "negative_triggers": list(),
            "entities": list()
        }

        # Construct the tokens of the document
        tokens = [token for sentence in one_data["sentences"] for token in sentence]
        # Assert only one event (trigger) exists in each document
        assert len(one_data["evt_triggers"]) == 1
        assert len(one_data["evt_triggers"][0][2]) == 1
        # Extract the event annotation of the document
        document["events"] = [{
            "type": one_data["evt_triggers"][0][2][0][0],
            "triggers": [{
                "id": "NA",
                "trigger_word": " ".join(tokens[one_data["evt_triggers"][0][0]:one_data["evt_triggers"][0][1] + 1]),
                "position": token_pos_to_char_pos(tokens,
                                                  [one_data["evt_triggers"][0][0], one_data["evt_triggers"][0][1] + 1]),
                "arguments": list()
            }]
        }]
        # Assert the offset of the trigger annotation is correct
        assert document["events"][0]["triggers"][0]["trigger_word"] \
               == document["text"][document["events"][0]["triggers"][0]["position"][0]:
                                   document["events"][0]["triggers"][0]["position"][1]]

        # Save all the applicable arguments of the event
        event_links = [evt_link[2] for evt_link in one_data["gold_evt_links"]]
        # Extract the entity and argument annotations within the document
        for ent_span in one_data["ent_spans"]:
            assert len(ent_span[2]) == 1
            # Extract the entity annotations within the document
            entity = {
                "type": re.split(r"\d+", ent_span[2][0][0])[2],
                "mentions": [{
                    "id": "NA",
                    "mention": " ".join(tokens[ent_span[0]:ent_span[1] + 1]),
                    "position": token_pos_to_char_pos(tokens,
                                                      [ent_span[0], ent_span[1] + 1])
                }]
            }
            # Assert the offset in the entity annotation is correct
            assert entity["mentions"][0]["mention"] \
                   == document["text"][entity["mentions"][0]["position"][0]:entity["mentions"][0]["position"][1]]
            document["entities"].append(entity)

            # Link the entity to the event annotation
            if ent_span[2][0][0] in event_links:
                document["events"][0]["triggers"][0]["arguments"].append({
                    "role": entity["type"],
                    "mentions": [{
                        "id": "NA",
                        "mention": entity["mentions"][0]["mention"],
                        "position": entity["mentions"][0]["position"]
                    }]
                })

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
    arg_parser = argparse.ArgumentParser(description="RAMS")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/original/RAMS_1.0")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/RAMS")
    args = arg_parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Read the RAMS datasets from the local path
    train_data, valid_data, test_data = read_data(args.data_dir)
    # Convert the RAMS datasets to the unified OmniEvent format
    train_documents, valid_documents, test_documents \
        = convert_to_unified(train_data), convert_to_unified(valid_data), convert_to_unified(test_data)

    # Generate label2id and role2id
    gen_label2id_and_role2id(train_documents + valid_documents + test_documents, args.save_dir)

    # Save the documents into jsonl files.
    train_documents = generate_negative_trigger(train_documents)
    json.dump(train_documents, open(os.path.join(args.save_dir, "train.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "train.unified.jsonl"), args.save_dir, train_documents)

    valid_documents = generate_negative_trigger(valid_documents)
    json.dump(valid_documents, open(os.path.join(args.save_dir, "valid.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "valid.unified.jsonl"), args.save_dir, valid_documents)

    test_documents = generate_negative_trigger(test_documents)
    json.dump(test_documents, open(os.path.join(args.save_dir, "test.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "test.unified.jsonl"), args.save_dir, test_documents)
