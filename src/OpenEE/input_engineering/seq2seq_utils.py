import os 
import re 
import pdb 
import json 
from collections import defaultdict


def prepare_output(arguments, template):
    """Prepare seq2seq output for an instance.
    
    Args:
        arguments: Arguments of a trigger. 
        template: The template of the event. 
    
    Returns:
        output: Output targets
    """
    re_template = re.compile("<.*?>")
    argument_markers = re_template.findall(template)
    output = []
    for marker in argument_markers:
        value = None 
        role = marker[1:-1]
        if role not in arguments:
            value = "NA"
        else:
            value = []
            for mention in arguments[role]:
                value.append(mention)
            value = " [SEP] ".join(value)
        output.append(marker + " " + value)
    return " ".join(output)


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
    if tokenizer.eos_token in output:
        end_idx = output.index(tokenizer.eos_token)
    else:
        print("Warning! No eos token in", output)
        end_idx = None 
    output = output[:end_idx]
    role_template = re.compile("(.*?)>")
    for span in output.split("<"):
        # role 
        roles = role_template.findall(span)
        if len(roles) != 1:
            print("Warning!", span)
        if len(roles) == 0:
            continue
        role = roles[roles[0].split("_")[-1]]
        # value 
        values = span.split(">")[-1]
        for value in values.split("[SEP]"):
            arguments[role].append(value.strip())
    return arguments


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




    
    
