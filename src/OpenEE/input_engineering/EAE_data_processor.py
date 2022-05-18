from operator import xor
import os 
import pdb 
import json
from numpy import sort
import torch 
import logging

from typing import List
from tqdm import tqdm 
from torch.utils.data import Dataset
from .input_utils import get_start_poses, check_if_start, get_word_position


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for event extractioin."""

    def __init__(self, example_id, text, trigger_left=None, trigger_right=None, argu_left=None, argu_right=None, labels=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            text: List of str. The untokenized text.
            triggerL: Left position of trigger.
            triggerR: Light position of tigger.
            labels: Event type of the trigger
        """
        self.example_id = example_id
        self.text = text
        self.trigger_left = trigger_left 
        self.trigger_right = trigger_right
        self.argu_left = argu_left
        self.argu_right = argu_right
        self.labels = labels


class InputFeatures(object):
    """Input features of an instance."""
    
    def __init__(self, 
                example_id, 
                input_ids, 
                attention_mask, 
                token_type_ids=None, 
                labels=None,
        ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels


class DataProcessor(Dataset):
    """Base class of data processor."""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def read_examples(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def _truncate(self, outputs, max_seq_length):
        is_truncation = False 
        if len(outputs["input_ids"]) > max_seq_length:
            print("An instance exceeds the maximum length.")
            is_truncation = True 
            for key in ["input_ids", "attention_mask", "token_type_ids", "offset_mapping"]:
                if key not in outputs:
                    continue
                outputs[key] = outputs[key][:max_seq_length]
        return outputs, is_truncation
    
    def get_ids(self):
        ids = []
        for example in self.examples:
            ids.append(example.example_id)
        return ids 

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        features = self.input_features[index]
        data_dict = dict(
            input_ids = torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask = torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict
        
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask", "token_type_ids", "trigger_left_mask", "trigger_right_mask"]:
            if key not in output_batch:
                continue
            output_batch[key] = output_batch[key][:, :input_length]
        if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
            if self.config.truncate_seq2seq_output:
                output_length = int((output_batch["labels"]!=-100).sum(-1).max())
                output_batch["labels"] = output_batch["labels"][:, :output_length]
            else:
                output_batch["labels"] = output_batch["labels"][:, :input_length] 
        return output_batch


class TCProcessor(DataProcessor):
    """Data processor for token classification."""

    def __init__(self, config, tokenizer, input_file):
        super().__init__(config, tokenizer)
        self.read_examples(input_file)
        self.convert_examples_to_features()


    def is_overlap(self, pos1, pos2):
        pos1_range = list(range(pos1[0], pos1[1]))
        pos2_range = list(range(pos2[0], pos2[1]))
        if pos1[0] in pos2_range or \
            pos1[1] in pos2_range or \
            pos2[0] in pos1_range or \
            pos2[1] in pos1_range:
            return True 
        return False 


    def read_examples(self, input_file):
        self.examples = []
        with open(input_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                # training and valid set
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        argu_for_trigger = set()
                        for argument in event["arguments"]:
                            for mention in argument["mentions"]:
                                example = InputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    trigger_left=trigger["position"][0],
                                    trigger_right=trigger["position"][1],
                                    argu_left=mention["position"][0],
                                    argu_right=mention["position"][1],
                                    labels=argument["role"]
                                )
                                if self.is_overlap(trigger["position"], mention["position"]):
                                    # raise ValueError("Overlap")
                                    print("Overlap")
                                argu_for_trigger.add(f"{mention['mention']}-{mention['position'][0]}-{mention['position'][1]}")
                                self.examples.append(example)
                        for entity in item["entities"]:
                            for mention in entity["mentions"]:
                                key = f"{mention['mention']}-{mention['position'][0]}-{mention['position'][1]}"
                                if key in argu_for_trigger:
                                    continue
                                example = InputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    trigger_left=trigger["position"][0],
                                    trigger_right=trigger["position"][1],
                                    argu_left=mention["position"][0],
                                    argu_right=mention["position"][1],
                                    labels="NA"
                                )
                                if self.is_overlap(trigger["position"], mention["position"]):
                                    continue 
                                self.examples.append(example)
                            
    
    def insert_marker(self, text, trigger_position, argument_position, markers, whitespace=True):
        space = " " if whitespace else ""
        # xx <event> trigger </event> xx xx <argument> argument </argument> xx .
        if trigger_position[1] <= argument_position[0]: 
            l_text = text[:trigger_position[0]]
            trigger = markers[0] + space + text[trigger_position[0]:trigger_position[1]] + space + markers[1]
            b_text = text[trigger_position[1]:argument_position[0]]
            argument = markers[2] + space + text[argument_position[0]:argument_position[1]] + space + markers[3]
            r_text = text[argument_position[1]:]
            text = l_text + trigger + b_text + argument + r_text
        else: # xx <argument> trigger </argument> xx xx <trigger> argument </trigger> xx .
            l_text = text[:argument_position[0]]
            argument = markers[2] + space + text[argument_position[0]:argument_position[1]] + space + markers[3]
            b_text = text[argument_position[1]:trigger_position[0]]
            trigger = markers[0] + space + text[trigger_position[0]:trigger_position[1]] + space + markers[1]
            r_text = text[trigger_position[1]:]
            text = l_text + argument + b_text + trigger + r_text
        return text 

    
    def convert_examples_to_features(self): 
        # merge and then tokenize
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for TC"):
            text = self.insert_marker(example.text, 
                                        [example.trigger_left, example.trigger_right], 
                                        [example.argu_left, example.argu_right], 
                                        self.config.markers, 
                                        True)
            outputs = self.tokenizer(text, 
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.max_seq_length)
            is_overflow = False 
            try:
                left = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[0]))
                right = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[1]))
            except: 
                logger.warning("Markers are not in the input tokens.")
                is_overflow = True
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
                
            features = InputFeatures(
                example_id = example.example_id,
                input_ids = outputs["input_ids"],
                attention_mask = outputs["attention_mask"],
                token_type_ids = outputs["token_type_ids"]
            )
            if example.labels is not None:
                features.labels = self.config.label2id[example.labels]
                if is_overflow:
                    features.labels = -100
            self.input_features.append(features)


            

            

