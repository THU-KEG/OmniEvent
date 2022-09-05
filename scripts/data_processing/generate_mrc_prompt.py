import json
import jsonlines
import os

from typing import Dict, List, Optional


def gen_prompts(input_folder: Optional[str] = "../../data/processed/ace2005-zh/",
                dump: Optional[bool] = False) -> Dict[str, List[str]]:
    """Generates prompts for the Machine Reading Comprehension (MRC) model.

    Generates prompts for the Machine Reading Comprehension (MRC) model. An event schema containing the argument roles
    of each event type is firstly constructed, and then the prompts for the MRC model are generated, following the
    following template:

        "{event type}_{argument_role}_not_implemented_yet}"

    Args:
        input_folder (`str`, `optional`, defaults to "../../data/processed/ace2005-zh/"):
            A string indicating the folder of the processed dataset.
        dump (`bool`, `optional`, defaults to `False`):
            A boolean variable indicating whether or not to dump the prompts into a csv file.

    Returns:
        A dictionary containing the argument roles existed in each event type.
    """
    event_schema = dict()
    for file in os.listdir(input_folder):
        if "unified" in file:
            data = list(jsonlines.open(os.path.join(input_folder, file)))
            for d in data:
                if "events" in d:
                    for event in d["events"]:
                        event_type = event["type"].replace(":", ".")   # manual fix for ace-oneie
                        if event_type not in event_schema:
                            event_schema[event_type] = []
                        for trigger in event["triggers"]:
                            if "arguments" in trigger:
                                for argument in trigger["arguments"]:
                                    arg_role = argument["role"]
                                    if arg_role not in event_schema[event_type]:
                                        event_schema[event_type].append(arg_role)
    if dump:
        with open(os.path.join(input_folder, "description_queries.csv"), "w", encoding="utf-8") as f:
            for event_type in event_schema:
                for arg_role in event_schema[event_type]:
                    f.write("{}_{},{}\n".format(event_type, arg_role, "not_implemented_yet"))
    return event_schema


if __name__ == "__main__":
    processed_path = "../../data/processed"
    event_schemas = dict()
    for dataset in os.listdir(processed_path):
        if dataset in ["ace2005-dygie", ".DS_Store"]:   # use QAEE
            continue
        event_schemas[f"{dataset}"] = gen_prompts(input_folder=os.path.join(processed_path, dataset), dump=True)
