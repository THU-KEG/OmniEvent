from email.policy import default
import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict


def get_maven_submission(preds, instance_ids, result_file):
    all_results = defaultdict(list)
    for i, pred in enumerate(preds):
        example_id, candidate_id = instance_ids[i].split("-")
        all_results[example_id].append({
            "id": candidate_id,
            "type_id": int(pred)
        })
    with open(result_file, "w") as f:
        for data_id in all_results.keys():
            format_result = dict(id=data_id, predictions=[])
            for candidate in all_results[data_id]:
                format_result["predictions"].append(candidate)
            f.write(json.dumps(format_result) + "\n")


# def get_maven_submission_sl(preds, labels, test_file, result_file, tokenizer):
