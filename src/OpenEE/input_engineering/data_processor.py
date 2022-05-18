from operator import xor
import os 
import pdb 
import json
from numpy import sort
import torch 
import logging

from tqdm import tqdm 
from torch.utils.data import Dataset
from .input_utils import get_start_poses, check_if_start, get_word_position


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for event extractioin."""

    def __init__(self, example_id, text, trigger_left=None, trigger_right=None, labels=None):
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
        self.labels = labels


class InputFeatures(object):
    """Input features of an instance."""
    
    def __init__(self, example_id, input_ids, attention_mask, token_type_ids=None, trigger_left_mask=None, trigger_right_mask=None, labels=None):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_left_mask = trigger_left_mask
        self.trigger_right_mask = trigger_right_mask
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
        if features.trigger_left_mask is not None:
            data_dict["trigger_left_mask"] = torch.tensor(features.trigger_left_mask, dtype=torch.float32)
        if features.trigger_right_mask is not None:
            data_dict["trigger_right_mask"] = torch.tensor(features.trigger_right_mask, dtype=torch.float32)
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

    def read_examples(self, input_file):
        self.examples = []
        with open(input_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                # training and valid set
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            example = InputExample(
                                example_id=trigger["id"],
                                text=item["text"],
                                trigger_left=trigger["position"][0],
                                trigger_right=trigger["position"][1],
                                labels=event["type"]
                            )
                            self.examples.append(example)
                if "negative_triggers" in item:
                    for neg in item["negative_triggers"]:
                        example = InputExample(
                            example_id=neg["id"],
                            text=item["text"],
                            trigger_left=neg["position"][0],
                            trigger_right=neg["position"][1],
                            labels="NA" 
                        )
                        self.examples.append(example)
                # test set 
                if "candidates" in item:
                    for candidate in item["candidates"]:
                        example = InputExample(
                            example_id=candidate["id"],
                            text=item["text"],
                            trigger_left=candidate["position"][0],
                            trigger_right=candidate["position"][1]
                        )
                        # # if test set has labels
                        # assert not (self.config.test_exists_labels ^ ("type" in candidate))
                        # if "type" in candidate:
                        #     example.labels = candidate["type"]
                        self.examples.append(example)

    def convert_examples_to_features(self): 
        # merge and then tokenize
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for TC"):
            text = example.text[:example.trigger_left] + self.config.markers[0] + " " \
                        + example.text[example.trigger_left:example.trigger_right] \
                        + " " + self.config.markers[1] + example.text[example.trigger_right:]
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
            # trigger position mask 
            left_mask = [1] * left + [0] * (self.config.max_seq_length-left)
            right_mask = [1] * right + [0]* (self.config.max_seq_length-right)
                
            features = InputFeatures(
                example_id = example.example_id,
                input_ids = outputs["input_ids"],
                attention_mask = outputs["attention_mask"],
                token_type_ids = outputs["token_type_ids"],
                trigger_left_mask = left_mask,
                trigger_right_mask = right_mask
            )
            if example.labels is not None:
                features.labels = self.config.label2id[example.labels]
                if is_overflow:
                    features.labels = -100
            self.input_features.append(features)


class SLProcessor(DataProcessor):
    """Data processor for sequence labeling."""

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
                    labels = ["O"] * len(item["text"].split())
                else:
                    labels = ["O"] * len(item["text"])
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            if self.config.language == "English": 
                                left_pos = len(item["text"][:trigger["position"][0]].split())
                                right_pos = len(item["text"][:trigger["position"][1]].split())
                            else:
                                assert self.config.language == "Chinese"
                                left_pos = trigger["position"][0]
                                right_pos = trigger["position"][1]
                            labels[left_pos] = f"B-{event['type']}"
                            for i in range(left_pos+1, right_pos):
                                labels[i] = f"I-{event['type']}"
                example = InputExample(
                    example_id = item["id"],
                    text = item["text"],
                    labels = labels
                )
                self.examples.append(example)
        
    def get_final_labels(self, example, outputs):
         # map subtoken to word
        start_poses = get_start_poses(example.text)
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
                    final_labels.append(self.config.label2id[example.labels[word_idx]])
                    last_word_idx = word_idx
        final_labels[0] = self.config.label2id["O"]
        return final_labels
    
    def get_final_labels_zh(self, example, outputs):
         # map subtoken to word
        start_poses = list(range(len(example.text)))
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
                    final_labels.append(self.config.label2id[example.labels[word_idx]])
                    last_word_idx = word_idx
        final_labels[0] = self.config.label2id["O"]
        return final_labels


    def convert_examples_to_features(self):
        self.input_features = []
        self.is_overflow = []
        for example in tqdm(self.examples, desc="Processing features for SL"):
            outputs = self.tokenizer(example.text,
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
                final_labels = self.get_final_labels(example, outputs)
            else:
                final_labels = self.get_final_labels_zh(example, outputs)
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

            

            

