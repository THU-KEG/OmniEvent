import argparse
import json
import jsonlines
import os

from typing import Dict, List, Tuple


def split(data: List[Dict],
          train_split: List[str],
          valid_split: List[str],
          test_split: List[str]) -> Tuple[List, List, List]:
    """Splits the ERE datasets into training, validation, and test sets.

    Splits the ERE datasets into training, validation, and test sets based on the pre-defined document lists for each
    set.

    Args:
        data (`List[Dict]`):
            A list of dictionaries representing the combination of the ERE datasets: LDC2015E29, LDC2015E68, and
            LDC2015E78.
        train_split (`List[str]`), valid_split (`List[str]`), test_split (`List[str]`):
            Three lists of strings including the document ids for each set.

    Returns:
        train_data (`List[Dict]`), valid_data (`List[Dict]`), test_data (`List[Dict]`):
            Three list of dictionaries representing the training, validation, and test sets of the ERE dataset.
    """
    # Initialize three lists for the training, validation, and test sets
    train_data, valid_data, test_data = list(), list(), list()

    # Append the data into the corresponding datasets
    for one_data in data:
        if one_data["id"].split("-")[0] in train_split:
            train_data.append(one_data)
        elif one_data["id"].split("-")[0] in valid_split:
            valid_data.append(one_data)
        elif one_data["id"].split("-")[0] in test_split:
            test_data.append(one_data)
        else:
            print("The document %s does not belong to any set." % one_data["id"])

    assert len(data) == len(train_data) + len(valid_data) + len(test_data)
    return train_data, valid_data, test_data


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
    arg_parser = argparse.ArgumentParser(description="ERE")
    arg_parser.add_argument("--ldc2015e29", type=str,
                            default="../../../data/processed/LDC2015E29/LDC2015E29.unified.jsonl")
    arg_parser.add_argument("--ldc2015e68", type=str,
                            default="../../../data/processed/LDC2015E68/LDC2015E68.unified.jsonl")
    arg_parser.add_argument("--ldc2015e78", type=str,
                            default="../../../data/processed/LDC2015E78/LDC2015E78.unified.jsonl")
    arg_parser.add_argument("--split_dir", type=str, default="./split")
    arg_parser.add_argument("--save_dir", type=str, default="../../../data/processed/ERE")
    args = arg_parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load the processed dataset from the local path
    data = list(jsonlines.open(args.ldc2015e29)) + list(jsonlines.open(args.ldc2015e68)) + \
           list(jsonlines.open(args.ldc2015e78))

    # Load the split data from the local path
    train_split = [line.strip("\n") for line in open(os.path.join(args.split_dir, "train.doc.txt")).readlines()]
    valid_split = [line.strip("\n") for line in open(os.path.join(args.split_dir, "dev.doc.txt")).readlines()]
    test_split = [line.strip("\n") for line in open(os.path.join(args.split_dir, "test.doc.txt")).readlines()]

    # Split the ERE dataset into training, validation, and test sets
    train_data, valid_data, test_data = split(data, train_split, valid_split, test_split)

    # Save the different sets of the ERE dataset to the path
    gen_label2id_and_role2id(train_data + valid_data + test_data, args.save_dir)
    json.dump(train_data, open(os.path.join(args.save_dir, "train.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "train.unified.jsonl"), args.save_dir, train_data)

    json.dump(valid_data, open(os.path.join(args.save_dir, "valid.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "valid.unified.jsonl"), args.save_dir, valid_data)

    json.dump(test_data, open(os.path.join(args.save_dir, "test.json"), "w"), indent=4)
    to_jsonl(os.path.join(args.save_dir, "test.unified.jsonl"), args.save_dir, test_data)
