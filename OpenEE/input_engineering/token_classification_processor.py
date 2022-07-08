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


class EDTCProcessor(EDDataProcessor):
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
                            example = EDInputExample(
                                example_id=trigger["id"],
                                text=item["text"],
                                trigger_left=trigger["position"][0],
                                trigger_right=trigger["position"][1],
                                labels=event["type"]
                            )
                            self.examples.append(example)
                if "negative_triggers" in item:
                    for neg in item["negative_triggers"]:
                        example = EDInputExample(
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
                        example = EDInputExample(
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
            text_left = example.text[:example.trigger_left]
            text_mid = example.text[example.trigger_left:example.trigger_right]
            text_right = example.text[example.trigger_right:]

            if self.config.language == "Chinese":
                text = text_left + self.config.markers[0] + text_mid + self.config.markers[1] + text_right
            else:
                text = text_left + self.config.markers[0] + " " + text_mid + " " + self.config.markers[1] + text_right

            outputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.config.max_seq_length)
            is_overflow = False
            try:
                left = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[0]))
                right = outputs["input_ids"].index(self.tokenizer.convert_tokens_to_ids(self.config.markers[1]))
            except:
                logger.warning("Markers are not in the input tokens.")
                left = self.config.max_seq_length
                is_overflow = True

            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])

            # trigger position mask
            left_mask = [1] * left + [0] * (self.config.max_seq_length - left)
            right_mask = [0] * left + [1] * (self.config.max_seq_length - left)

            features = EDInputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                trigger_left_mask=left_mask,
                trigger_right_mask=right_mask
            )
            if example.labels is not None:
                features.labels = self.config.type2id[example.labels]
            self.input_features.append(features)


class EAETCProcessor(EAEDataProcessor):
    """Data processor for token classification."""

    def __init__(self, config, tokenizer, input_file, pred_file, is_training=False):
        super().__init__(config, tokenizer, pred_file, is_training)
        self.read_examples(input_file)
        self.convert_examples_to_features()

    def read_examples(self, input_file):
        self.examples = []
        trigger_idx = 0
        with open(input_file, "r") as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines, desc="Reading from %s" % input_file):
                item = json.loads(line.strip())
                # training and valid set
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        argu_for_trigger = set()
                        if self.event_preds is not None \
                            and not self.config.golden_trigger \
                            and not self.is_training:    
                            pred_event_type = self.event_preds[trigger_idx] 
                        else:
                            pred_event_type = event["type"]
                        for argument in trigger["arguments"]:
                            for mention in argument["mentions"]:
                                example = EAEInputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    pred_type=pred_event_type,
                                    true_type=event["type"],
                                    trigger_left=trigger["position"][0],
                                    trigger_right=trigger["position"][1],
                                    argu_left=mention["position"][0],
                                    argu_right=mention["position"][1],
                                    labels=argument["role"]
                                )
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
                                example = EAEInputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    pred_type=pred_event_type,
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
                
            features = EAEInputFeatures(
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
