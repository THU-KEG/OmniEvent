
import json
import logging

from tqdm import tqdm 
from collections import defaultdict
from .base_processor import (
    EAEDataProcessor,
    EAEInputExample,
    EAEInputFeatures
)


logger = logging.getLogger(__name__)


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
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        if self.event_preds is not None \
                            and not self.config.golden_trigger \
                            and not self.is_training:    
                            pred_event_type = self.event_preds[trigger_idx] 
                        else:
                            pred_event_type = event["type"]
                        if pred_event_type == "NA":
                            trigger_idx += 1
                            continue
                        labels = []
                        arguments_per_trigger = defaultdict(list)
                        for argument in trigger["arguments"]:
                            for mention in argument["mentions"]:
                                arguments_per_trigger[argument["role"]].append(mention["mention"])
                                labels.append(f"<extra_id_0> {argument['role']} {mention['mention']} <extra_id_1>")
                        if len(labels) != 0:
                            labels = "".join(labels)
                            labels = "<extra_id_0>" + labels + "<extra_id_1>"
                        else:       # no arguments for the trigger
                            labels = "<extra_id_0><extra_id_1>"
                        self.data_for_evaluation["golden_arguments"].append(dict(arguments_per_trigger))
                        example = EAEInputExample(
                            example_id=trigger["id"],
                            text=item["text"],
                            pred_type=pred_event_type,
                            true_type=event["type"],
                            trigger_left=trigger["position"][0],
                            trigger_right=trigger["position"][1],
                            labels=labels
                        )
                        trigger_idx += 1
                        self.examples.append(example)
                # negative triggers 
                for neg in item["negative_triggers"]:
                    trigger_idx += 1


    def insert_marker(self, text, type, trigger_pos, markers, whitespace=True):
        space = " " if whitespace else ""
        markered_text = ""
        tokens = text.split()
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos == trigger_pos[0]:
                markered_text += markers[type][0] + space
            char_pos += len(token) + len(space)
            markered_text += token + space
            if char_pos == trigger_pos[1] + len(space):
                markered_text += markers[type][1] + space
        markered_text = markered_text.strip()
        return markered_text
        
    def convert_examples_to_features(self):
        self.input_features = []
        whitespace = True if self.config.language == "English" else False 
        for example in tqdm(self.examples, desc="Processing features for SL"):
            # context 
            text = self.insert_marker(example.text, 
                                      example.true_type, 
                                      [example.trigger_left, example.trigger_right],
                                      self.config.markers,
                                      whitespace)
            input_context = self.tokenizer(text,
                                           truncation=True,
                                           padding="max_length",
                                           max_length=self.config.max_seq_length)
            # output labels
            label_outputs = self.tokenizer(example.labels,
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.config.max_out_length)
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