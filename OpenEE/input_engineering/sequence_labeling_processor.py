import json
import logging

from tqdm import tqdm
from .base_processor import (
    EDDataProcessor,
    EDInputExample,
    EDInputFeatures,
    EAEDataProcessor,
    EAEInputExample,
    EAEInputFeatures
)

logger = logging.getLogger(__name__)


class EDSLProcessor(EDDataProcessor):
    """Data processor for sequence labeling."""

    def __init__(self, config, tokenizer, input_file):
        super().__init__(config, tokenizer)
        self.read_examples(input_file)
        self.is_overflow = []
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

                labels = ["O"] * len(words)

                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            if self.config.language == "English":
                                left_pos = len(item["text"][:trigger["position"][0]].split())
                                right_pos = len(item["text"][:trigger["position"][1]].split())
                            elif self.config.language == "Chinese":
                                left_pos = trigger["position"][0]
                                right_pos = trigger["position"][1]
                            else:
                                raise NotImplementedError

                            labels[left_pos] = f"B-{event['type']}"
                            for i in range(left_pos + 1, right_pos):
                                labels[i] = f"I-{event['type']}"
                example = EDInputExample(
                    example_id=item["id"],
                    text=words,
                    labels=labels
                )
                self.examples.append(example)

    def get_final_labels(self, example, word_ids_of_each_token, label_all_tokens=False):
        final_labels = []
        pre_word_id = None
        for word_id in word_ids_of_each_token:
            if word_id is None:
                final_labels.append(-100)
            elif word_id != pre_word_id:  # first split token of a word
                final_labels.append(self.config.type2id[example.labels[word_id]])
            else:
                final_labels.append(self.config.type2id[example.labels[word_id]] if label_all_tokens else -100)
            pre_word_id = word_id

        return final_labels

    def convert_examples_to_features(self):
        self.input_features = []

        for example in tqdm(self.examples, desc="Processing features for SL"):
            outputs = self.tokenizer(example.text,
                                     padding="max_length",
                                     truncation=False,
                                     max_length=self.config.max_seq_length,
                                     is_split_into_words=True)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            outputs, is_overflow = self._truncate(outputs, self.config.max_seq_length)
            self.is_overflow.append(is_overflow)

            word_ids_of_each_token = outputs.word_ids()[: self.config.max_seq_length]
            final_labels = self.get_final_labels(example, word_ids_of_each_token, label_all_tokens=False)

            features = EDInputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                labels=final_labels
            )
            self.input_features.append(features)


class EAESLProcessor(EAEDataProcessor):
    """Data processor for sequence labeling."""

    def __init__(self, config, tokenizer, input_file, pred_file, is_training=False):
        super().__init__(config, tokenizer, pred_file, is_training)
        self.positive_candidate_indices = []
        self.is_overflow = []
        self.config.role2id["X"] = -100
        self.read_examples(input_file)
        self.convert_examples_to_features()

    def read_examples(self, input_file):
        self.examples = []
        trigger_idx = 0
        with open(input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())

                if self.config.language == "English":
                    words = item["text"].split()
                elif self.config.language == "Chinese":
                    words = list(item["text"])
                else:
                    raise NotImplementedError

                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            true_type = event["type"]
                            if self.is_training or self.config.golden_trigger or self.event_preds is None:
                                pred_type = true_type
                            else:
                                pred_type = self.event_preds[trigger_idx]

                            if self.config.language == "English":
                                trigger_left = len(item["text"][:trigger["position"][0]].split())
                                trigger_right = len(item["text"][:trigger["position"][1]].split())
                            elif self.config.language == "Chinese":
                                trigger_left = trigger["position"][0]
                                trigger_right = trigger["position"][1]
                            else:
                                raise NotImplementedError
                            labels = ["O"] * len(words)

                            for argument in trigger["arguments"]:
                                for mention in argument["mentions"]:
                                    if self.config.language == "English":
                                        left_pos = len(item["text"][:mention["position"][0]].split())
                                        right_pos = len(item["text"][:mention["position"][1]].split())
                                    elif self.config.language == "Chinese":
                                        left_pos = mention["position"][0]
                                        right_pos = mention["position"][1]
                                    else:
                                        raise NotImplementedError

                                    labels[left_pos] = f"B-{argument['role']}"
                                    for i in range(left_pos + 1, right_pos):
                                        labels[i] = f"I-{argument['role']}"

                            example = EAEInputExample(
                                example_id=item["id"],
                                text=words,
                                pred_type=pred_type,
                                true_type=event["type"],
                                trigger_left=trigger_left,
                                trigger_right=trigger_right,
                                labels=labels,
                            )
                            trigger_idx += 1

                            if self.eval_mode in ['default', 'loose']:
                                if pred_type == "NA":
                                    continue
                            self.examples.append(example)

                    # negative triggers
                    for trigger in item["negative_triggers"]:
                        true_type = "NA"
                        if self.eval_mode in ['default', 'strict'] and not self.is_training:  # loose mode has no neg
                            pred_type = self.event_preds[trigger_idx]
                            if pred_type != "NA":
                                if self.config.language == "English":
                                    trigger_left = len(item["text"][:trigger["position"][0]].split())
                                    trigger_right = len(item["text"][:trigger["position"][1]].split())
                                elif self.config.language == "Chinese":
                                    trigger_left = trigger["position"][0]
                                    trigger_right = trigger["position"][1]
                                else:
                                    raise NotImplementedError
                                labels = ["O"] * len(words)

                                example = EAEInputExample(
                                    example_id=item["id"],
                                    text=words,
                                    pred_type=pred_type,
                                    true_type=true_type,
                                    trigger_left=trigger_left,
                                    trigger_right=trigger_right,
                                    labels=labels,
                                )
                                self.examples.append(example)

                        trigger_idx += 1
                else:
                    for candi in item["candidates"]:
                        if self.config.language == "English":
                            trigger_left = len(item["text"][:candi["position"][0]].split())
                            trigger_right = len(item["text"][:candi["position"][1]].split())
                        elif self.config.language == "Chinese":
                            trigger_left = candi["position"][0]
                            trigger_right = candi["position"][1]
                        else:
                            raise NotImplementedError
                        labels = ["O"] * len(words)

                        pred_type = self.event_preds[trigger_idx]
                        if pred_type != "NA":
                            example = EAEInputExample(
                                example_id=item["id"],
                                text=words,
                                pred_type=pred_type,
                                true_type="NA",  # true type not given, set to NA.
                                trigger_left=trigger_left,
                                trigger_right=trigger_right,
                                labels=labels,
                            )
                            self.examples.append(example)
                            self.positive_candidate_indices.append(trigger_idx)

                        trigger_idx += 1

    def get_final_labels(self, labels, word_ids_of_each_token, label_all_tokens=False):
        final_labels = []
        pre_word_id = None
        for word_id in word_ids_of_each_token:
            if word_id is None:
                final_labels.append(-100)
            elif word_id != pre_word_id:  # first split token of a word
                final_labels.append(self.config.role2id[labels[word_id]])
            else:
                final_labels.append(self.config.role2id[labels[word_id]] if label_all_tokens else -100)
            pre_word_id = word_id

        return final_labels

    @staticmethod
    def insert_marker(text, event_type, labels, trigger_pos, markers):
        left, right = trigger_pos

        marked_text = text[:left] + [markers[event_type][0]] + text[left:right] + [markers[event_type][1]] + text[
                                                                                                             right:]
        marked_labels = labels[:left] + ["X"] + labels[left:right] + ["X"] + labels[right:]

        assert len(marked_text) == len(marked_labels)
        return marked_text, marked_labels

    def convert_examples_to_features(self):
        self.input_features = []
        self.is_overflow = []

        for example in tqdm(self.examples, desc="Processing features for SL"):
            text, labels = self.insert_marker(example.text,
                                              example.pred_type,
                                              example.labels,
                                              [example.trigger_left, example.trigger_right],
                                              self.config.markers)
            outputs = self.tokenizer(text,
                                     padding="max_length",
                                     truncation=False,
                                     max_length=self.config.max_seq_length,
                                     is_split_into_words=True)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            outputs, is_overflow = self._truncate(outputs, self.config.max_seq_length)
            self.is_overflow.append(is_overflow)

            word_ids_of_each_token = outputs.word_ids()[: self.config.max_seq_length]
            final_labels = self.get_final_labels(labels, word_ids_of_each_token, label_all_tokens=False)

            features = EAEInputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                labels=final_labels
            )
            self.input_features.append(features)
