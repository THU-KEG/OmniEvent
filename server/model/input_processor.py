import os 
import json 
import torch 

from collections import defaultdict
from typing import List 


def get_words(text: str,
              language: str) -> List[str]:
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
            language = "Chinese"
        else:
            language = "English"
        words = get_words(schema+text, language)
        input_context = self.tokenizer(words,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.max_seq_length,
                                       is_split_into_words=True)
        return dict(
            input_ids=torch.tensor(input_context["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(input_context["attention_mask"], dtype=torch.float32)
        )


    def tokenize(self, texts, schemas):
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
            output_batch[key] = output_batch[key][:, :input_length].cuda()
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
        text = self.insert_marker(text, trigger["offset"])
        if schema in ["<duee>", "<fewfc>", "<leven>"]:
            language = "Chinese"
        else:
            language = "English"
        words = get_words(schema+text, language)
        input_context = self.tokenizer(words,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.max_seq_length,
                                       is_split_into_words=True)
        return dict(
            input_ids=torch.tensor(input_context["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(input_context["attention_mask"], dtype=torch.float32)
        )

    def tokenize(self, instances):
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
            output_batch[key] = output_batch[key][:, :input_length].cuda()
        return output_batch
