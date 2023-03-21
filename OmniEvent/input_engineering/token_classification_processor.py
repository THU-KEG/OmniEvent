import json
import logging

from tqdm import tqdm
from typing import List, Optional, Dict

from .input_utils import check_is_argument, get_negative_argument_candidates, get_word_ids, char_pos_to_word_pos
from .base_processor import (
    EDDataProcessor,
    EDInputExample,
    EDInputFeatures,
    EAEDataProcessor,
    EAEInputExample,
    EAEInputFeatures,
)

logger = logging.getLogger(__name__)


class EDTCProcessor(EDDataProcessor):
    """Data processor for token classification for event detection.

    Data processor for token classification for event detection. The class is inherited from the`EDDataProcessor` class,
    in which the undefined functions, including `read_examples()` and `convert_examples_to_features()` are  implemented;
    the rest of the attributes and functions are multiplexed from the `EDDataProcessor` class.
    """

    def __init__(self,
                 config,
                 tokenizer: str,
                 input_file: str) -> None:
        """Constructs an EDTCProcessor."""
        super().__init__(config, tokenizer)
        self.read_examples(input_file)
        self.convert_examples_to_features()

    def read_examples(self,
                      input_file: str) -> None:
        """Obtains a collection of `EDInputExample`s for the dataset."""
        self.examples = []
        with open(input_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                # training and valid set
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            example = EDInputExample(
                                example_id=trigger["id"],
                                text=item["text"],
                                trigger_left=trigger["position"][0],
                                trigger_right=trigger["position"][1],
                                labels=event["type"],
                            )
                            self.examples.append(example)
                if "negative_triggers" in item:
                    for neg in item["negative_triggers"]:
                        example = EDInputExample(
                            example_id=neg["id"],
                            text=item["text"],
                            trigger_left=neg["position"][0],
                            trigger_right=neg["position"][1],
                            labels="NA",
                        )
                        self.examples.append(example)
                # test set 
                if "candidates" in item:
                    for candidate in item["candidates"]:
                        example = EDInputExample(
                            example_id=candidate["id"],
                            text=item["text"],
                            trigger_left=candidate["position"][0],
                            trigger_right=candidate["position"][1],
                            labels="NA",
                        )
                        self.examples.append(example)

    def convert_examples_to_features(self) -> None:
        """Converts the `EDInputExample`s into `EDInputFeatures`s."""
        # merge and then tokenize
        self.input_features = []
        for example in tqdm(self.examples, desc="Processing features for TC"):
            if self.config.insert_marker:
                text_left = example.text[:example.trigger_left]
                text_mid = example.text[example.trigger_left:example.trigger_right]
                text_right = example.text[example.trigger_right:]

                if self.config.language == "Chinese":
                    text = text_left + self.config.markers[0] + text_mid + self.config.markers[1] + text_right
                else:
                    text = text_left + self.config.markers[0] + " " + text_mid + " " + self.config.markers[1] + text_right

                outputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.config.max_seq_length)
                try:
                    left = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[0]))
                    right = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[1]))
                except:
                    logger.warning("Markers are not in the input tokens.")
                    left, right = 0, 0
            else:
                tokens = example.text.split()
                outputs = self.tokenizer(tokens, 
                                        padding="max_length", 
                                        truncation=True, 
                                        max_length=self.config.max_seq_length,
                                        is_split_into_words=True,
                                        add_special_tokens=True)
                trigger_word_left, trigger_word_right = char_pos_to_word_pos(example.text, 
                                                                        [example.trigger_left, example.trigger_right])
                word_ids_of_each_token = get_word_ids(self.tokenizer, outputs, tokens)[: self.config.max_seq_length]
                left, right = -1, -1
                for i, word_id in enumerate(word_ids_of_each_token):
                    if word_id == trigger_word_left and left == -1:
                        left = i
                    if word_id == trigger_word_right-1:
                        right = i

                if left == -1:
                    logger.warning("Overflow! %s" % example.text)
                if right == -1:
                    right = self.config.max_seq_length - 1
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])

            features = EDInputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                trigger_left=left,
                trigger_right=right,
            )
            if example.labels is not None:
                features.labels = self.config.type2id[example.labels]
            if left == -1:
                left = 0
                features.labels = -100
            self.input_features.append(features)


class EAETCProcessor(EAEDataProcessor):
    """Data processor for token classification for event argument extraction.

    Data processor for token classification for event argument extraction. The class is inherited from the
    `EAEDataProcessor` class, in which the undefined functions, including `read_examples()` and
    `convert_examples_to_features()` are  implemented; a new function entitled `insert_marker()` is defined, and
    the rest of the attributes and functions are multiplexed from the `EAEDataProcessor` class.
    """

    def __init__(self,
                 config,
                 tokenizer: str,
                 input_file: str,
                 pred_file: str,
                 is_training: Optional[bool] = False):
        """Constructs a `EAETCProcessor`."""
        super().__init__(config, tokenizer, pred_file, is_training)
        self.read_examples(input_file)
        self.convert_examples_to_features()

    def read_examples(self,
                      input_file: str) -> None:
        """Obtains a collection of `EAEInputExample`s for the dataset."""
        self.examples = []
        trigger_idx = 0
        with open(input_file, "r") as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines, desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                if "events" in item:
                    for event in item["events"]:
                        for trigger in event["triggers"]:
                            pred_type = self.get_single_pred(trigger_idx, input_file, true_type=event["type"])
                            trigger_idx += 1

                            if self.config.eae_eval_mode in ['default', 'loose'] and pred_type == "NA":
                                continue

                            positive_offsets = []
                            for argument in trigger["arguments"]:
                                for mention in argument["mentions"]:
                                    example = EAEInputExample(
                                        example_id=trigger["id"],
                                        text=item["text"],
                                        pred_type=pred_type,
                                        true_type=event["type"],
                                        trigger_left=trigger["position"][0],
                                        trigger_right=trigger["position"][1],
                                        argument_left=mention["position"][0],
                                        argument_right=mention["position"][1],
                                        labels=argument["role"],
                                    )
                                    positive_offsets.append(mention["position"])
                                    self.examples.append(example)

                            self.add_negative_arguments(item=item, trigger=trigger, pred_type=pred_type,
                                                        true_type=event["type"], positive_offsets=positive_offsets)

                    # negative triggers
                    for trigger in item["negative_triggers"]:
                        if self.config.eae_eval_mode in ['default', 'strict']:
                            pred_type = self.get_single_pred(trigger_idx, input_file, true_type="NA")
                            if pred_type != "NA":
                                self.add_negative_arguments(item=item, trigger=trigger, pred_type=pred_type,
                                                            true_type="NA")
                        trigger_idx += 1

                if "candidates" in item:
                    for candi in item["candidates"]:
                        pred_type = self.event_preds[trigger_idx]   # we can only use pred type here, gold not available
                        if pred_type != "NA":
                            self.add_negative_arguments(item=item, trigger=candi, pred_type=pred_type, true_type="NA")
                        trigger_idx += 1

            if self.event_preds is not None:
                assert trigger_idx == len(self.event_preds)

    @staticmethod
    def insert_marker(text: str,
                      type: str,
                      trigger_position: List[int],
                      argument_position: List[int],
                      markers: Dict[str, str],
                      whitespace: Optional[bool] = True) -> str:
        """Adds a marker at the start and end position of event triggers and argument mentions."""
        marked_text = ""
        for i, char in enumerate(text):
            if i == trigger_position[0]:
                marked_text += markers[type][0]
                marked_text += " " if whitespace else ""
            if i == argument_position[0]:
                marked_text += markers["argument"][0]
                marked_text += " " if whitespace else ""
            marked_text += char
            if i == trigger_position[1] - 1:
                marked_text += " " if whitespace else ""
                marked_text += markers[type][1]
            if i == argument_position[1] - 1:
                marked_text += " " if whitespace else ""
                marked_text += markers["argument"][1]
        return marked_text

    def add_negative_arguments(self, item, trigger, pred_type, true_type, positive_offsets=None):
        neg_arg_candidates = get_negative_argument_candidates(item, positive_offsets=positive_offsets)

        for mention in neg_arg_candidates:
            is_argument = check_is_argument(mention, positive_offsets)
            if not is_argument:
                # negative arguments
                example = EAEInputExample(
                    example_id=trigger["id"],
                    text=item["text"],
                    pred_type=pred_type,
                    true_type=true_type,
                    trigger_left=trigger["position"][0],
                    trigger_right=trigger["position"][1],
                    argument_left=mention["position"][0],
                    argument_right=mention["position"][1],
                    labels="NA",
                )
                self.examples.append(example)

    def convert_examples_to_features(self) -> None:
        """Converts the `EAEInputExample`s into `EAEInputFeatures`s."""
        # merge and then tokenize
        self.input_features = []
        whitespace = True if self.config.language == "English" else False
        for example in tqdm(self.examples, desc="Processing features for TC"):
            if self.config.insert_marker:
                text = self.insert_marker(example.text,
                                        example.pred_type,
                                        [example.trigger_left, example.trigger_right],
                                        [example.argument_left, example.argument_right],
                                        self.config.markers,
                                        whitespace)
                outputs = self.tokenizer(text,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.config.max_seq_length)
                is_overflow = False
                # argument position 
                try:
                    argument_left = outputs["input_ids"].index(
                        self.tokenizer.convert_tokens_to_ids(self.config.markers["argument"][0]))
                    argument_right = outputs["input_ids"].index(
                        self.tokenizer.convert_tokens_to_ids(self.config.markers["argument"][1]))
                except:
                    argument_left, argument_right = 0, 0
                    logger.warning("Argument markers are not in the input tokens.")
                    is_overflow = True
                # trigger position
                try:
                    trigger_left = outputs["input_ids"].index(
                        self.tokenizer.convert_tokens_to_ids(self.config.markers[example.pred_type][0]))
                    trigger_right = outputs["input_ids"].index(
                        self.tokenizer.convert_tokens_to_ids(self.config.markers[example.pred_type][1]))
                except:
                    trigger_left, trigger_right = 0, 0
                    logger.warning("Trigger markers are not in the input tokens.")
            else:
                tokens = example.text.split()
                outputs = self.tokenizer(tokens, 
                                        padding="max_length", 
                                        truncation=True, 
                                        max_length=self.config.max_seq_length,
                                        is_split_into_words=True,
                                        add_special_tokens=True)
                word_ids_of_each_token = get_word_ids(self.tokenizer, outputs, tokens)[: self.config.max_seq_length]
                trigger_word_left, trigger_word_right = char_pos_to_word_pos(example.text, 
                                                                        [example.trigger_left, example.trigger_right])
                argument_word_left, argument_word_right = char_pos_to_word_pos(example.text, 
                                                                        [example.argument_left, example.argument_right])
                trigger_left, trigger_right = -1, -1
                argument_left, argument_right = -1, -1
                for i, word_id in enumerate(word_ids_of_each_token):
                    if word_id == trigger_word_left and trigger_left == -1:
                        trigger_left = i
                    if word_id == trigger_word_right-1:
                        trigger_right = i
                    if word_id == argument_word_left and argument_left == -1:
                        argument_left = i
                    if word_id == argument_word_right-1:
                        argument_right = i
                
                if trigger_left == -1 or argument_left == -1:
                    logger.warning("Overflow! %s" % example.text)
                if trigger_right == -1:
                    trigger_right = self.config.max_seq_length - 1
                if argument_right == -1:
                    argument_right = self.config.max_seq_length - 1

            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            
            if self.config.consider_event_type:
                token_type_ids = [0] * len(outputs["input_ids"])
                for idx in range(trigger_left, trigger_right+1):
                    token_type_ids[idx] = self.config.type2id[example.pred_type]

                for idx in range(argument_left, argument_right):
                    token_type_ids[idx] = self.config.type2id[example.pred_type]

                outputs["token_type_ids"] = token_type_ids

            features = EAEInputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                trigger_left=trigger_left,
                trigger_right=trigger_right,
                argument_left=argument_left,
                argument_right=argument_right,
            )
            if example.labels is not None:
                features.labels = self.config.role2id[example.labels]
                if trigger_left == -1:
                    trigger_left = 0
                    features.labels = -100
                if argument_left == -1:
                    argument_left = 0
                    features.labels = -100
            self.input_features.append(features)
