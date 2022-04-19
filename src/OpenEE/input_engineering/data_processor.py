from lib2to3.pgen2.tokenize import tokenize
from operator import xor
import os 
import pdb 
import json
import torch 
import logging

from tqdm import tqdm 
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for event extractioin."""

    def __init__(self, example_id, sentence, trigger_left, trigger_right, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            sentence: List of str. The untokenized sentence.
            triggerL: Left position of trigger.
            triggerR: Light position of tigger.
            label: Event type of the trigger
        """
        self.example_id = example_id
        self.sentence = sentence
        self.trigger_left = trigger_left 
        self.trigger_right = trigger_right
        self.label = label


class InputFeatures(object):
    """Input features of an instance."""
    
    def __init__(self, example_id, input_ids, input_mask, segment_ids, trigger_left, trigger_right, label=None):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.label = label


class DataProcessor(Dataset):
    """Base class of data processor."""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    

    def read_examples(self, input_file):
        raise NotImplementedError


    def convert_examples_to_features(self):
        raise NotImplementedError
    

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
            input_mask = torch.tensor(features.input_mask, dtype=torch.float32),
            segment_ids = torch.tensor(features.segment_ids, dtype=torch.long),
            trigger_left = torch.tensor(features.trigger_left, dtype=torch.long),
            trigger_right = torch.tensor(features.trigger_right, dtype=torch.long),
        )
        if features.label is not None:
            data_dict["labels"] = torch.tensor(features.label, dtype=torch.long)
        return data_dict
        
    
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        input_length = int(output_batch["input_mask"].sum(-1).max())
        for key in ["input_ids", "input_mask", "segment_ids"]:
            output_batch[key] = output_batch[key][:, :input_length]
        if len(output_batch["labels"].shape) == 2:
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
                        for mention in event["mentions"]:
                            example = InputExample(
                                example_id=mention["id"],
                                sentence=item["sentence"],
                                trigger_left=mention["position"][0],
                                trigger_right=mention["position"][1],
                                label=event["type"]
                            )
                            self.examples.append(example)
                if "negative_triggers" in item:
                    for neg in item["negative_triggers"]:
                        example = InputExample(
                            example_id=neg["id"],
                            sentence=item["sentence"],
                            trigger_left=neg["position"][0],
                            trigger_right=neg["position"][1],
                            label="NA" 
                        )
                        self.examples.append(example)
                # test set 
                if "candidates" in item:
                    for candidate in item["candidates"]:
                        example = InputExample(
                            example_id=candidate["id"],
                            sentence=item["sentence"],
                            trigger_left=candidate["position"][0],
                            trigger_right=candidate["position"][1]
                        )
                        # if test set has labels
                        assert not (self.config.test_exists_labels ^ ("type" in candidate))
                        if "type" in candidate:
                            example.label = candidate["type"]
                        self.examples.append(example)

        
    def convert_examples_to_features(self): 
        # merge and then tokenize
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for TC"):
            sentence = example.sentence[:example.trigger_left] + self.config.markers[0] + " " \
                        + example.sentence[example.trigger_left:example.trigger_right] \
                        + " " + self.config.markers[1] + example.sentence[example.trigger_right:]
            outputs = self.tokenizer(sentence, 
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.max_seq_length)
            try:
                left = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[0]))
                right = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[1]))
            except:
                logger.warning("Markers are not in the input tokens. %s", sentence)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
                
            features = InputFeatures(
                example_id = example.example_id,
                input_ids = outputs["input_ids"],
                input_mask = outputs["attention_mask"],
                segment_ids = outputs["token_type_ids"],
                trigger_left = left,
                trigger_right = right
            )
            # pdb.set_trace()
            if example.label is not None:
                features.label = self.config.label2id[example.label]
            self.input_features.append(features)


class SLProcessor(DataProcessor):
    """Data processor for sequence labeling."""

    def __init__(self, config, tokenizer, input_file):
        super().__init__(config, tokenizer)
        self.read_examples(input_file)
        self.convert_examples_to_features()

    
    def read_examples(self, input_file):
        self.examples = []
        with open(input_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                label = ["O"] * len(item["sentence"].split())
                if "events" in item:
                    for event in item["events"]:
                        for mention in event["mentions"]:
                            left_pos = len(item["sentence"][:mention["position"][0]].split())
                            right_pos = len(item["sentence"][:mention["position"][1]].split())
                            label[left_pos] = f"B-{event['type']}"
                            for i in range(left_pos+1, right_pos):
                                label[i] = f"I-{event['type']}"
                example = InputExample(
                    example_id = item["id"],
                    sentence = item["sentence"],
                    trigger_left = -1,
                    trigger_right= -1,
                    label = label
                )
                self.examples.append(example)
        

    def convert_examples_to_features(self):
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for SL"):
            outputs = self.tokenizer(example.sentence,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.config.max_seq_length,
                                    return_offsets_mapping=True)
            # pdb.set_trace()
            final_labels = []
            for i, offset in enumerate(outputs["offset_mapping"]):
                if outputs["attention_mask"][i] == 0:
                    final_labels.append(-100)
                    continue
                if offset[0] == 0 and offset[1] == 0:
                    final_labels.append(0) # the id of "O" is 0
                    continue
                pos = len(example.sentence[:offset[1]].split()) - 1
                final_labels.append(self.config.label2id[example.label[pos]])
            # pdb.set_trace()
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            features = InputFeatures(
                example_id = example.example_id,
                input_ids = outputs["input_ids"],
                input_mask = outputs["attention_mask"],
                segment_ids = outputs["token_type_ids"],
                trigger_left = -1,
                trigger_right = -1,
                label = final_labels
            )
            self.input_features.append(features)
            




        
            
                    
                

