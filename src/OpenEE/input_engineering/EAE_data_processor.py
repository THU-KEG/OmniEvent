from cProfile import label
from operator import xor
import os 
import pdb 
import json
from re import L
from string import whitespace
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

    def __init__(self, example_id, text, pred_type, true_type, trigger_left=None, trigger_right=None, argu_left=None, argu_right=None, labels=None):
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
        self.pred_type = pred_type
        self.true_type = true_type
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
        self.config.role2id["X"] = -100
    
    def read_examples(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def get_pred_types(self):
        pred_types = []
        for example in self.examples:
            pred_types.append(example.pred_type)
        return pred_types 

    def get_true_types(self):
        true_types = []
        for example in self.examples:
            true_types.append(example.true_type)
        return true_types

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

    def __init__(self, config, tokenizer, input_file, pred_file):
        super().__init__(config, tokenizer)
        self.read_examples(input_file, pred_file)
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


    def read_examples(self, input_file, pred_file):
        self.examples = []
        trigger_idx = 0
        preds = json.load(open(pred_file))
        with open(input_file, "r") as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines, desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                # training and valid set
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        argu_for_trigger = set()
                        for argument in trigger["arguments"]:
                            for mention in argument["mentions"]:
                                example = InputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    pred_type=preds[trigger_idx],
                                    true_type=event["type"],
                                    trigger_left=trigger["position"][0],
                                    trigger_right=trigger["position"][1],
                                    argu_left=mention["position"][0],
                                    argu_right=mention["position"][1],
                                    labels=argument["role"]
                                )
                                if "train" in input_file or self.config.golden_trigger:
                                    example.pred_type = event["type"]
                                argu_for_trigger.add(mention['mention_id'])
                                self.examples.append(example)
                        for entity in item["entities"]:
                            # check whether the entity is an argument 
                            is_argument = False 
                            for mention in entity["mentions"]:
                                if mention["mention_id"] in argu_for_trigger:
                                    is_argument = True 
                                    break 
                            if is_argument:
                                continue
                            # negative arguments 
                            for mention in entity["mentions"]:
                                example = InputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    pred_type=preds[trigger_idx],
                                    true_type=event["type"],
                                    trigger_left=trigger["position"][0],
                                    trigger_right=trigger["position"][1],
                                    argu_left=mention["position"][0],
                                    argu_right=mention["position"][1],
                                    labels="NA"
                                )
                                if "train" in input_file or self.config.golden_trigger:
                                    example.pred_type = event["type"]
                                self.examples.append(example)
                        trigger_idx += 1
                # negative triggers 
                for neg in item["negative_triggers"]:
                    trigger_idx += 1         
    
    def insert_marker(self, text, type, trigger_position, argument_position, markers, whitespace=True):
        markered_text = ""
        for i, char in enumerate(text):
            if i == trigger_position[0]:
                markered_text += markers[type][0]
                markered_text += " " if whitespace else ""
            if i == argument_position[0]:
                markered_text += markers["argument"][0]
                markered_text += " " if whitespace else ""
            markered_text += char 
            if i == trigger_position[1]-1:
                markered_text += " " if whitespace else ""
                markered_text += markers[type][1]
            if i ==argument_position[1]-1:
                markered_text += " " if whitespace else ""
                markered_text += markers["argument"][1]
        return markered_text

    def convert_examples_to_features(self): 
        # merge and then tokenize
        self.input_features = []
        whitespace = True if self.config.language == "English" else False 
        for example in tqdm(self.examples, desc="Processing features for TC"):
            text = self.insert_marker(example.text, 
                                        example.pred_type,
                                        [example.trigger_left, example.trigger_right], 
                                        [example.argu_left, example.argu_right], 
                                        self.config.markers, 
                                        whitespace)
            outputs = self.tokenizer(text, 
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.max_seq_length)
            is_overflow = False 
            try:
                left = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers["argument"][0]))
                right = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers["argument"][1]))
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
                features.labels = self.config.role2id[example.labels]
                if is_overflow:
                    features.labels = -100
            self.input_features.append(features)


class SLProcessor(DataProcessor):
    """Data processor for sequence labeling."""

    def __init__(self, config, tokenizer, input_file, pred_file):
        super().__init__(config, tokenizer)
        self.read_examples(input_file, pred_file)
        self.convert_examples_to_features()

    def read_examples(self, input_file, pred_file):
        self.examples = []
        trigger_idx = 0
        preds = json.load(open(pred_file))
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        if self.config.language == "English":
                            labels = ["O"] * len(item["text"].split())
                        else:
                            labels = ["O"] * len(item["text"])
                        for argument in trigger["arguments"]:
                            for mention in argument["mentions"]:
                                if self.config.language == "English": 
                                    left_pos = len(item["text"][:mention["position"][0]].split())
                                    right_pos = len(item["text"][:mention["position"][1]].split())
                                else:
                                    assert self.config.language == "Chinese"
                                    left_pos = mention["position"][0]
                                    right_pos = mention["position"][1]
                                labels[left_pos] = f"B-{argument['role']}"
                                for i in range(left_pos+1, right_pos):
                                    labels[i] = f"I-{argument['role']}"
                        example = InputExample(
                            example_id = item["id"],
                            text = item["text"],
                            pred_type=preds[trigger_idx],
                            true_type=event["type"],
                            trigger_left=trigger["position"][0],
                            trigger_right=trigger["position"][1],
                            labels = labels
                        )
                        if "train" in input_file or self.config.golden_trigger:
                            example.pred_type = event["type"]
                        trigger_idx += 1
                        self.examples.append(example)
                # negative triggers 
                for neg in item["negative_triggers"]:
                    trigger_idx += 1   
        
    def get_final_labels(self, text, labels, outputs):
         # map subtoken to word
        start_poses = get_start_poses(text)
        subtoken2word = []
        current_word_idx = None
        for offset in outputs["offset_mapping"]:
            if offset[0] == offset[1]:
                subtoken2word.append(-1)
            else:
                if check_if_start(start_poses, offset):
                    current_word_idx = get_word_position(start_poses, offset)
                    subtoken2word.append(current_word_idx)
                else:
                    subtoken2word.append(current_word_idx)
        # mapping word labels to subtoken labels
        final_labels = []
        last_word_idx = None 
        for word_idx in subtoken2word:
            if word_idx == -1:
                final_labels.append(-100)
            else:
                if word_idx == last_word_idx: # subtoken
                    final_labels.append(-100)
                else:  # new word
                    final_labels.append(self.config.role2id[labels[word_idx]])
                    last_word_idx = word_idx
        return final_labels
    
    def get_final_labels_zh(self, text, labels, outputs):
         # map subtoken to word
        start_poses = list(range(len(text)))
        subtoken2word = []
        current_word_idx = None
        for offset in outputs["offset_mapping"]:
            if offset[0] == offset[1]:
                subtoken2word.append(-1)
            else:
                current_word_idx = get_word_position(start_poses, offset)
                subtoken2word.append(current_word_idx)
        # mapping word labels to subtoken labels
        final_labels = []
        last_word_idx = None 
        for word_idx in subtoken2word:
            if word_idx == -1:
                final_labels.append(-100)
            else:
                if word_idx == last_word_idx: # subtoken
                    final_labels.append(-100)
                else:  # new word
                    final_labels.append(self.config.role2id[labels[word_idx]])
                    last_word_idx = word_idx
        return final_labels
    

    def insert_marker(self, text, type, labels, trigger_pos, markers, whitespace=True):
        space = " " if whitespace else ""
        markered_text = ""
        markered_labels = []
        tokens = text.split()
        assert len(tokens) == len(labels)
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos == trigger_pos[0]:
                markered_text += markers[type][0] + space
                markered_labels.append("X")
            char_pos += len(token) + len(space)
            markered_text += token + space
            markered_labels.append(labels[i])
            if char_pos == trigger_pos[1] + len(space):
                markered_text += markers[type][1] + space
                markered_labels.append("X")
        markered_text = markered_text.strip()
        # pdb.set_trace()
        assert len(markered_text.split(space)) == len(markered_labels)
        return markered_text, markered_labels


    def convert_examples_to_features(self):
        self.input_features = []
        self.is_overflow = []
        whitespace = True if self.config.language == "English" else False 
        for example in tqdm(self.examples, desc="Processing features for SL"):
            text, labels = self.insert_marker(example.text, 
                                              example.pred_type, 
                                              example.labels,
                                              [example.trigger_left, example.trigger_right],
                                              self.config.markers,
                                              whitespace)
            outputs = self.tokenizer(text,
                                    padding="max_length",
                                    truncation=False,
                                    max_length=self.config.max_seq_length,
                                    return_offsets_mapping=True)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            outputs, is_overflow = self._truncate(outputs, self.config.max_seq_length)
            self.is_overflow.append(is_overflow)
            if self.config.language == "English":
                final_labels = self.get_final_labels(text, labels, outputs)
            else:
                final_labels = self.get_final_labels_zh(text, labels, outputs)
            features = InputFeatures(
                example_id = example.example_id,
                input_ids = outputs["input_ids"],
                attention_mask = outputs["attention_mask"],
                token_type_ids = outputs["token_type_ids"],
                labels = final_labels
            )
            self.input_features.append(features)


class Seq2SeqProcessor(DataProcessor):
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
                labels = []
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            labels.append({
                                "type": event["type"],
                                "word": trigger["trigger_word"],
                                "position": trigger["position"]
                            })
                example = InputExample(
                    example_id = item["id"],
                    text = item["text"],
                    trigger_left = -1,
                    trigger_right = -1,
                    labels = labels
                )
                self.examples.append(example)
        
    def convert_examples_to_features(self):
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for SL"):
            outputs = self.tokenizer(example.text,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.max_seq_length,
                                    return_offsets_mapping=True)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            labels = ""
            for mention_label in example.labels:
                if outputs["offset_mapping"][-1][0] != 0 and \
                    outputs["offset_mapping"][-1][1] != 0 and \
                    mention_label["position"][1] > outputs["offset_mapping"][-1][1]:
                    continue
                labels += f"{mention_label['type']}:{mention_label['word']};"
            label_outputs = self.tokenizer(labels,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.max_out_length)
            # set -100 to unused token 
            for i, flag in enumerate(label_outputs["attention_mask"]):
                if flag == 0:
                    label_outputs["input_ids"][i] = -100
            features = InputFeatures(
                example_id = example.example_id,
                input_ids = outputs["input_ids"],
                attention_mask = outputs["attention_mask"],
                token_type_ids = outputs["token_type_ids"],
                labels = label_outputs["input_ids"]
            )
            self.input_features.append(features)
            


            

            

