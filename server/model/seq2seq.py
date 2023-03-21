import re 
import os 
import json 
import torch 
from typing import List, Tuple, Union

from transformers import MT5ForConditionalGeneration, MT5TokenizerFast

from .input_processor import EDProcessor, EAEProcessor
from .constraint_decoding import get_constraint_decoder


type_start = "<"
type_end = ">"
split_word = ":"

def get_backbone(
        model_type, 
        model_name_or_path, 
        tokenizer_name, 
        markers: list=[],
        new_tokens:list = []
    ):
    model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
    tokenizer = MT5TokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config


def extract_argument(raw_text: str,
                     instance_id: Union[int, str],
                     template=re.compile(f"{type_start}|{type_end}")) -> List[Tuple]:
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
    # constraint_decoder = get_constraint_decoder(tokenizer=tokenizer,
    #                                             source_prefix=None)
    def prefix_allowed_tokens_fn(batch_id, sent):
        src_sentence = inputs['input_ids'][batch_id]
        return constraint_decoder.constraint_decoding(batch_id=batch_id,
                                                            src_sentence=src_sentence,
                                                            tgt_generated=sent)
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


def do_event_detection(model, tokenizer, texts, schemas):
    data_processor = EDProcessor(tokenizer)
    inputs = data_processor.tokenize(texts, schemas)
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


def do_event_argument_extraction(model, tokenizer, instances):
    data_processor = EAEProcessor(tokenizer)
    inputs = data_processor.tokenize(instances)
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

    