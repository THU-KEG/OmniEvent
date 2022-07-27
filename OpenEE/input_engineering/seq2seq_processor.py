import re
import json
import logging

from tqdm import tqdm 
from collections import defaultdict
from .base_processor import (
    EDInputExample,
    EDDataProcessor,
    EDInputFeatures,
    EAEDataProcessor,
    EAEInputExample,
    EAEInputFeatures
)

type_start = "<"
type_end = ">"
split_word = ":"

logger = logging.getLogger(__name__)


def extract_argument(raw_text, instance_id, event_type, template=re.compile(f"[{type_start}{type_end}]")):
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
            arguments.append((instance_id, event_type, role, value))
    arguments = list(set(arguments))
    return arguments


class EDSeq2SeqProcessor(EDDataProcessor):
    "Data processor for sequence to sequence."

    def __init__(self, config, tokenizer, input_file):
        super().__init__(config, tokenizer)
        self.read_examples(input_file)
        self.convert_examples_to_features()
    
    def read_examples(self, input_file):
        self.examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                if self.config.language == "English":
                    words = item["text"].split()
                elif self.config.language == "Chinese":
                    words = list(item["text"])
                else:
                    raise NotImplementedError
                # training and valid set
                if "events" in item:
                    labels = []
                    for event in item["events"]:
                        type = "".join("".join(event["type"].split(".")[-1].split("-")).split("_")).lower()
                        for trigger in event["triggers"]:
                            labels.append(f"{type_start} {type}{split_word} {trigger['trigger_word']} {type_end}")
                    if len(labels) != 0:
                        labels = "".join(labels)
                        # labels = type_start + labels + type_end 
                    else:       # no arguments for the trigger
                        labels = ""
                        # labels = f"{type_start}{type_end}"
                    example = EDInputExample(
                        example_id=item["id"],
                        text=words,
                        labels=labels
                    )
                    self.examples.append(example)

    def convert_examples_to_features(self):
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for SL"):
            # context 
            input_context = self.tokenizer(example.text,
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_seq_length,
                                           is_split_into_words=True)
            # output labels
            label_outputs = self.tokenizer(example.labels.split(),
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_out_length,
                                           is_split_into_words=True)
            # set -100 to unused token 
            for i, flag in enumerate(label_outputs["attention_mask"]):
                if flag == 0:
                    label_outputs["input_ids"][i] = -100
            features = EDInputFeatures(
                example_id = example.example_id,
                input_ids = input_context["input_ids"],
                attention_mask = input_context["attention_mask"],
                labels = label_outputs["input_ids"]
            )
            self.input_features.append(features)


class EAESeq2SeqProcessor(EAEDataProcessor):
    "Data processor for sequence to sequence."

    def __init__(self, config, tokenizer, input_file, pred_file, is_training=False):
        super().__init__(config, tokenizer, pred_file, is_training)
        self.read_examples(input_file)
        self.convert_examples_to_features()
    
    def read_examples(self, input_file):
        self.examples = []
        self.data_for_evaluation["golden_arguments"] = []
        self.data_for_evaluation["roles"] = []
        trigger_idx = 0
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                prefix = []
                if self.config.language == "English":
                    words = prefix + item["text"].split()
                    whitespace = " "
                elif self.config.language == "Chinese":
                    words = prefix + list(item["text"])
                    whitespace = ""
                else:
                    raise NotImplementedError
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            if self.event_preds is not None \
                                and not self.config.golden_trigger \
                                and not self.is_training:    
                                pred_event_type = self.event_preds[trigger_idx] 
                            else:
                                pred_event_type = event["type"]
                            trigger_idx += 1
                            # Evaluation mode for EAE
                            # If the predicted event type is NA, We don't consider the trigger
                            if self.config.eae_eval_mode in ["default", "loose"]\
                                and pred_event_type == "NA":
                                continue
                            labels = []
                            arguments_per_trigger = defaultdict(list)
                            for argument in trigger["arguments"]:
                                role = argument["role"]
                                for mention in argument["mentions"]:
                                    arguments_per_trigger[argument["role"]].append(mention["mention"])
                                    labels.append(f"{type_start}{whitespace}{role}{split_word}{whitespace}{mention['mention']}{whitespace}{type_end}")
                            if len(labels) != 0:
                                labels = "".join(labels)
                            else:       # no arguments for the trigger
                                labels = ""
                            self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                            example = EAEInputExample(
                                example_id=trigger["id"],
                                text=words,
                                pred_type=pred_event_type,
                                true_type=event["type"],
                                trigger_left=trigger["position"][0],
                                trigger_right=trigger["position"][1],
                                labels=labels
                            )
                            self.examples.append(example)
                    # negative triggers 
                    for neg_trigger in item["negative_triggers"]:
                        if self.event_preds is not None \
                            and not self.config.golden_trigger \
                            and not self.is_training:    
                            pred_event_type = self.event_preds[trigger_idx]
                        else:
                            pred_event_type = "NA"
                        trigger_idx += 1         
                        if self.config.eae_eval_mode == "loose":
                            continue
                        elif self.config.eae_eval_mode in ["default", "strict"]:
                            if pred_event_type != "NA":
                                labels = ""
                                arguments_per_trigger = {}
                                self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                                example = EAEInputExample(
                                    example_id=trigger_idx-1,
                                    text=words,
                                    pred_type=pred_event_type,
                                    true_type="NA",
                                    trigger_left=neg_trigger["position"][0],
                                    trigger_right=neg_trigger["position"][1],
                                    labels=labels
                                )
                                self.examples.append(example)
                        else:
                            raise ValueError("Invaild eac_eval_mode: %s" % self.config.eae_eval_mode)
                else:
                    for candi in item["candidates"]:
                        pred_event_type = self.event_preds[trigger_idx]
                        trigger_idx += 1
                        if pred_event_type != "NA":
                            labels = ""
                            # labels = f"{type_start}{type_end}"
                            arguments_per_trigger = {}
                            self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                            example = EAEInputExample(
                                example_id=item["id"],
                                text=words,
                                pred_type=pred_event_type,
                                true_type="NA",   # true type not given, set to NA.
                                trigger_left=candi["position"][0],
                                trigger_right=candi["position"][1],
                                labels=labels,
                            )
                            self.examples.append(example)
            if self.event_preds is not None:
                assert trigger_idx == len(self.event_preds)


    def insert_marker(self, tokens, trigger_pos, markers, whitespace=True):
        space = " " if whitespace else ""
        markered_words = []
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos == trigger_pos[0]:
                markered_words.append(markers[0])
            char_pos += len(token) + len(space)
            markered_words.append(token)
            if char_pos == trigger_pos[1] + len(space):
                markered_words.append(markers[1])
        return markered_words
        

    def convert_examples_to_features(self):
        self.input_features = []
        whitespace = True if self.config.language == "English" else False 
        for example in tqdm(self.examples, desc="Processing features for SL"):
            # context 
            words = self.insert_marker(example.text, 
                                      [example.trigger_left, example.trigger_right],
                                      self.config.markers,
                                      whitespace)
            input_context = self.tokenizer(words,
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_seq_length,
                                           is_split_into_words=True)
            # output labels
            label_outputs = self.tokenizer(example.labels.split(),
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.config.max_out_length,
                                           is_split_into_words=True)
            # set -100 to unused token 
            for i, flag in enumerate(label_outputs["attention_mask"]):
                if flag == 0:
                    label_outputs["input_ids"][i] = -100
            features = EAEInputFeatures(
                example_id = example.example_id,
                input_ids = input_context["input_ids"],
                attention_mask = input_context["attention_mask"],
                labels = label_outputs["input_ids"]
            )
            self.input_features.append(features)