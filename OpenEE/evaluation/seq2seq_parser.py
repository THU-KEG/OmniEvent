import os 
import re
import pdb 
import json 


def extract_argument(raw_text, instance_id, event_type, template=re.compile(r"[<>]")):
    arguments = []
    for span in template.split(raw_text):
        if span.strip() == "":
            continue
        words = span.strip().split()
        role = words[0]
        value = " ".join(words[1:])
        if value.strip() != "":
            arguments.append((instance_id, event_type, role, value))
    arguments = list(set(arguments))
    return arguments


