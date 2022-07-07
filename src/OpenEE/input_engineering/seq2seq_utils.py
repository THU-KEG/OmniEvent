import os 
import re 
import pdb 
import json 
from collections import defaultdict


def prepare_output(arguments, roles, template):
    """Prepare seq2seq output for an instance.
    
    Args:
        arguments: Arguments of a trigger. 
        roles: Roles of arguments
        template: The template of the event. 
    
    Returns:
        output: Output targets
    """
    re_template = re.compile("<extra_id_(\d+)>")
    role_ids = re_template.findall(template)
    output = []
    for id in role_ids:
        role = roles[int(id)]
        value = None 
        if role not in arguments:
            value = ""
        else:
            value = []
            for mention in arguments[role]:
                value.append(mention)
            value = " [SEP] ".join(value)
        output.append(f"<extra_id_{id}>" + value)
    return "".join(output)


def decode_arguments(output, roles, tokenizer):
    """Decode arguments from output text.

    Arg:
        output: Plain output text.
        roles: Roles for the instance.
        tokenizer: Tokenizer.
    
    Returns:
        arguments: Arguments
    """
    arguments = defaultdict(list)
    role_template = re.compile("extra_id_(\d+)>")
    for span in output.split("<"):
        # role 
        role_ids = role_template.findall(span)
        if len(role_ids) == 0:
            continue
        if int(role_ids[0]) >= len(roles):
            continue
        role = roles[int(role_ids[0])]
        # value 
        values = span.split(">")[-1]
        for value in values.split("[SEP]"):
            if value.strip() != "":
                arguments[role].append(value.strip())
    for key in arguments:
        arguments[key] = list(set(arguments[key]))
    return dict(arguments)


def get_final_preds_labels(preds, labels):
    final_preds = []
    final_labels = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        for key in pred:
            for value in pred[key]:
                final_preds.append(f"{i}-{key}-{value}")
        for key in label:
            for value in label[key]:
                final_labels.append(f"{i}-{key}-{value}")
    return final_preds, final_labels

# def get_final_preds_labels(preds, labels):
#     final_preds = []
#     final_labels = []
#     for i, (pred, label) in enumerate(zip(preds, labels)):
#         for key in pred:
#             for value in pred[key]:
#                 final_preds.append(key)
#                 if key in label and value in label[key]:
#                     final_labels.append(key)
#                 else:
#                     final_labels.append("NA")
#         for key in label:
#             for value in label[key]:
#                 if key in pred and value in pred[key]:
#                     continue
#                 final_labels.append(key)
#                 final_preds.append("NA")
#     return final_preds, final_labels



    
    
