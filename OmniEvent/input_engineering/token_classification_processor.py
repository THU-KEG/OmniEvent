import json
import logging

from tqdm import tqdm
from typing import List, Optional, Dict

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
        markered_text = ""
        for i, char in enumerate(text):
            if i == trigger_position[0]:
                markered_text += markers[type][0]
                markered_text += " " if whitespace else ""
            if i == argument_position[0]:
                markered_text += markers["argument"][0]
                markered_text += " " if whitespace else ""
            markered_text += char
            if i == trigger_position[1] - 1:
                markered_text += " " if whitespace else ""
                markered_text += markers[type][1]
            if i == argument_position[1] - 1:
                markered_text += " " if whitespace else ""
                markered_text += markers["argument"][1]
        return markered_text

    def convert_examples_to_features(self) -> None:
        """Converts the `EAEInputExample`s into `EAEInputFeatures`s."""
        # merge and then tokenize
        self.input_features = []
        whitespace = True if self.config.language == "English" else False
        for example in tqdm(self.examples, desc="Processing features for TC"):
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
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])

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
                if is_overflow:
                    features.labels = -100
            self.input_features.append(features)
