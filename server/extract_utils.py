import os 
import json 
from io_format import Result, Event, Argument


def find_position(mention, text):
    char_start = text.index(mention)
    char_end = char_start + len(mention)
    return [char_start, char_end]


def get_ed_result(texts, triggers):
    results = []
    for i, text in enumerate(texts):
        triggers_in_text = [trigger for trigger in triggers if trigger[0]==i]
        result = Result()
        events = []
        for trigger in triggers_in_text:
            type = trigger[1]
            mention = trigger[2]
            if mention not in text:
                continue
            offset = find_position(mention, text)
            event = {
                "type": type,
                "trigger": mention,
                "offset": offset
            }
            events.append(event)
        results.append({
            "text": text,
            "events": events 
        })
    return results


def get_eae_result(instances, arguments):
    results = []
    for i, instance in enumerate(instances):
        result = Result()
        events = []
        for trigger, argus_in_trigger in zip(instance["triggers"], arguments):
            event = Event()
            event_arguments = []
            for argu in argus_in_trigger:
                role = argu[1]
                mention = argu[2]
                if mention not in instance["text"]:
                    continue
                offset = find_position(mention, instance["text"])
                argument = {
                    "mention": mention,
                    "offset": offset,
                    "role": role
                }
                event_arguments.append(argument)
            events.append({
                "type": trigger["type"] if "type" in trigger else "NA",
                "offset": trigger["offset"], 
                "trigger": trigger["mention"],
                "arguments": event_arguments
            })
        results.append({
            "text": instance["text"],
            "events": events
        })
    return results
        

def prepare_for_eae_from_input(texts, all_triggers, schemas):
    instances = []
    for text, triggers, schema in zip(texts, all_triggers, schemas):
        instance = {
            "text": text,
            "schema": schema,
            "triggers": []
        }
        for trigger in triggers:
            instance["triggers"].append({
                "mention": trigger[0],
                "offset": [trigger[1], trigger[2]]
            })
        instances.append(instance)
    return instances


def prepare_for_eae_from_pred(texts, triggers, schemas):
    instances = []
    for i, text in enumerate(texts):
        triggers_in_text = [trigger for trigger in triggers if trigger[0]==i]
        instance = {
            "text": text,
            "schema": schemas[i],
            "triggers": []
        }
        for trigger in triggers_in_text:
            type = trigger[1]
            mention = trigger[2]
            if mention not in text:
                continue
            offset = find_position(mention, text)
            instance["triggers"].append({
                "type": type, 
                "mention": mention,
                "offset": offset
            })
        instances.append(instance)
    return instances 
