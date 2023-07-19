import re 
import torch 

from collections import defaultdict
from .io_format import Result, Event


split_word = ":"

def get_words(text, language):
    if language == "English":
        words = text.split()
    elif language == "Chinese":
        words = list(text)
    else:
        raise NotImplementedError
    return words


class EDProcessor():
    def __init__(self, tokenizer, max_seq_length=160):
        self.tokenizer = tokenizer 
        self.max_seq_length = max_seq_length

    def tokenize_per_instance(self, text, schema):
        if schema in ["<duee>", "<fewfc>", "<leven>"]:
            words = get_words(schema+text, "Chinese")
        else:
            words = get_words(schema+text, "English")
        input_context = self.tokenizer(words,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.max_seq_length,
                                       is_split_into_words=True)
        return dict(
            input_ids=torch.tensor(input_context["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(input_context["attention_mask"], dtype=torch.float32)
        )


    def tokenize(self, texts, schemas, device):
        batch = []
        for text, schema in zip(texts, schemas):
            batch.append(self.tokenize_per_instance(text, schema))
        # output batch features 
        output_batch = defaultdict(list)
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        # truncate
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key not in output_batch:
                continue
            output_batch[key] = output_batch[key][:, :input_length].to(device)
        return output_batch


class EAEProcessor():
    def __init__(self, tokenizer, max_seq_length=160):
        self.tokenizer = tokenizer 
        self.max_seq_length = max_seq_length

    def insert_marker(self, text, trigger_pos, whitespace=True):
        space = " " if whitespace else ""
        markered_text = ""
        tokens = text.split()
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos == trigger_pos[0]:
                markered_text += "<event>" + space
            char_pos += len(token) + len(space)
            markered_text += token + space
            if char_pos == trigger_pos[1] + len(space):
                markered_text += "</event>" + space
        markered_text = markered_text.strip()
        return markered_text

    def tokenize_per_instance(self, text, trigger, schema):
        if schema in ["<duee>", "<fewfc>", "<leven>"]:
            language = "Chinese"
        else:
            language = "English"
        whitespace = True if language == "English" else False
        text = self.insert_marker(text, trigger["offset"], whitespace)
        words = get_words(text, language)
        input_context = self.tokenizer(words,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.max_seq_length,
                                       is_split_into_words=True)
        return dict(
            input_ids=torch.tensor(input_context["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(input_context["attention_mask"], dtype=torch.float32)
        )

    def tokenize(self, instances, device):
        batch = []
        for i, instance in enumerate(instances):
            for trigger in instance["triggers"]:
                batch.append(self.tokenize_per_instance(instance["text"], trigger, instance["schema"]))
        # output batch features 
        output_batch = defaultdict(list)
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        # truncate
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key not in output_batch:
                continue
            output_batch[key] = output_batch[key][:, :input_length].to(device)
        return output_batch


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


def extract_argument(raw_text, instance_id, template=re.compile(r"<|>")):
    arguments = []
    for span in template.split(raw_text):
        if span.strip() == "":
            continue
        words = span.strip().split(split_word)
        if len(words) != 2:
            continue
        role = words[0].strip().replace(" ", "")
        value = words[1].strip().replace(" ", "")
        if role != "" and value != "":
            arguments.append((instance_id, role, value))
    arguments = list(set(arguments))
    return arguments


def generate(model, tokenizer, inputs):
    gen_kwargs = {
        "max_length": 128,
        "num_beams": 4,
        "synced_gpus": False,
        "prefix_allowed_tokens_fn": None 
    }

    if "attention_mask" in inputs:
        gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

    generation_inputs = inputs["input_ids"]

    generated_tokens = model.generate(
        generation_inputs,
        **gen_kwargs,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)


def do_event_detection(model, tokenizer, texts, schemas, device):
    data_processor = EDProcessor(tokenizer)
    inputs = data_processor.tokenize(texts, schemas, device)
    decoded_preds = generate(model, tokenizer, inputs)
    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    pred_triggers = []
    for i, pred in enumerate(decoded_preds):
        pred = clean_str(pred)
        pred_triggers.extend(extract_argument(pred, i))
    return pred_triggers


def do_event_argument_extraction(model, tokenizer, instances, device="cuda"):
    data_processor = EAEProcessor(tokenizer)
    inputs = data_processor.tokenize(instances, device)
    decoded_preds = generate(model, tokenizer, inputs)
    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    pred_triggers = []
    for i, pred in enumerate(decoded_preds):
        pred = clean_str(pred)
        pred_triggers.append(extract_argument(pred, i))
    return pred_triggers

    