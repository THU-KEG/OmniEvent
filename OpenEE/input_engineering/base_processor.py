import os
import json
import torch
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EDInputExample(object):
    """A single training/test example for event extraction."""

    def __init__(self,
                 example_id,
                 text,
                 trigger_left=None,
                 trigger_right=None,
                 labels=None):
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


class EDInputFeatures(object):
    """Input features of an instance."""

    def __init__(self,
                 example_id,
                 input_ids,
                 attention_mask,
                 token_type_ids=None,
                 trigger_left=None,
                 trigger_right=None,
                 labels=None):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.labels = labels


class EAEInputExample(object):
    """A single training/test example for event extraction."""

    def __init__(self, example_id, text, pred_type, true_type,
                 input_template=None,
                 trigger_left=None,
                 trigger_right=None,
                 argument_role=None,
                 argument_left=None,
                 argument_right=None,
                 labels=None):
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
        self.input_template = input_template
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.argument_role = argument_role
        self.argument_left = argument_left
        self.argument_right = argument_right
        self.labels = labels


class EAEInputFeatures(object):
    """Input features of an instance."""

    def __init__(self,
                 example_id,
                 input_ids,
                 attention_mask,
                 token_type_ids=None,
                 trigger_left=None,
                 trigger_right=None,
                 argument_left=None,
                 argument_right=None,
                 labels=None,
                 ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.argument_left = argument_left
        self.argument_right = argument_right
        self.labels = labels


class EDDataProcessor(Dataset):
    """Base class of data processor for event detection."""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.examples = []
        self.input_features = []

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
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.trigger_left is not None:
            data_dict["trigger_left"] = torch.tensor(features.trigger_left, dtype=torch.float32)
        if features.trigger_right is not None:
            data_dict["trigger_right"] = torch.tensor(features.trigger_right, dtype=torch.float32)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key not in output_batch:
                    continue
                output_batch[key] = output_batch[key][:, :input_length]
            if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
                if self.config.truncate_seq2seq_output:
                    output_length = int((output_batch["labels"] != -100).sum(-1).max())
                    output_batch["labels"] = output_batch["labels"][:, :output_length]
                else:
                    output_batch["labels"] = output_batch["labels"][:, :input_length]
        return output_batch


class EAEDataProcessor(Dataset):
    """Base class of data processor."""

    def __init__(self, config, tokenizer, pred_file, is_training):
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        if hasattr(config, "role2id"):
            self.config.role2id["X"] = -100
        self.examples = []
        self.input_features = []
        # data for trainer evaluation 
        self.data_for_evaluation = {}
        # event prediction file path 
        if pred_file is not None:
            if not os.path.exists(pred_file):
                logger.warning("%s doesn't exist.We use golden triggers" % pred_file)
                self.event_preds = None
            else:
                self.event_preds = json.load(open(pred_file))
        else:
            logger.warning("Event predictions is none! We use golden triggers.")
            self.event_preds = None

    def read_examples(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def get_data_for_evaluation(self):
        self.data_for_evaluation["pred_types"] = self.get_pred_types()
        self.data_for_evaluation["true_types"] = self.get_true_types()
        self.data_for_evaluation["ids"] = self.get_ids()
        if self.examples[0].argument_role is not None:
            self.data_for_evaluation["roles"] = self.get_roles()
        return self.data_for_evaluation

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
    
    def get_roles(self):
        roles = []
        for example in self.examples:
            roles.append(example.argument_role)
        return roles 

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
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.trigger_left is not None:
            data_dict["trigger_left"] = torch.tensor(features.trigger_left, dtype=torch.long)
        if features.trigger_right is not None:
            data_dict["trigger_right"] = torch.tensor(features.trigger_right, dtype=torch.long)
        if features.argument_left is not None:
            data_dict["argument_left"] = torch.tensor(features.argument_left, dtype=torch.long)
        if features.argument_right is not None:
            data_dict["argument_right"] = torch.tensor(features.argument_right, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key not in output_batch:
                    continue
                output_batch[key] = output_batch[key][:, :input_length]
            if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
                if self.config.truncate_seq2seq_output:
                    output_length = int((output_batch["labels"] != -100).sum(-1).max())
                    output_batch["labels"] = output_batch["labels"][:, :output_length]
                else:
                    output_batch["labels"] = output_batch["labels"][:, :input_length]
        return output_batch
