
import json
import logging

from tqdm import tqdm 
from .base_processor import (
    EAEDataProcessor,
    EAEInputExample,
    EAEInputFeatures
)
from .mrc_converter import read_query_templates


logger = logging.getLogger(__name__)


class EAEMRCProcessor(EAEDataProcessor):
    "Data processor for machine reading comprehension."

    def __init__(self, config, tokenizer, input_file, pred_file, is_training=False):
        super().__init__(config, tokenizer, pred_file, is_training)
        self.read_examples(input_file)
        self.convert_examples_to_features()

    def read_examples(self, input_file):
        self.examples = []
        self.data_for_evaluation["golden_arguments"] = []
        trigger_idx = 0
        query_templates = read_query_templates(self.config.prompt_file)
        template_id = 3
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
                        trigger_left = len(item["text"][:trigger["position"][0]].split())
                        trigger_right = len(item["text"][:trigger["position"][1]].split())
                        for role in query_templates[pred_event_type].keys():
                            query = query_templates[pred_event_type][role][template_id]
                            query = query.replace("[trigger]", trigger["trigger_word"])
                            if self.is_training:
                                no_answer = True 
                                for argument in trigger["arguments"]:
                                    if argument["role"] != role:
                                        continue
                                    no_answer = False
                                    for mention in argument["mentions"]:
                                        left_pos = len(item["text"][:mention["position"][0]].split())
                                        right_pos = len(item["text"][:mention["position"][1]].split())
                                        example = EAEInputExample(
                                            example_id=trigger["id"],
                                            text=item["text"],
                                            pred_type=pred_event_type,
                                            true_type=event["type"],
                                            input_template=query,
                                            trigger_left=trigger_left,
                                            trigger_right=trigger_right,
                                            argu_left=left_pos,
                                            argu_right=right_pos-1
                                        )
                                        self.examples.append(example)
                                if no_answer:
                                    example = EAEInputExample(
                                        example_id=trigger["id"],
                                        text=item["text"],
                                        pred_type=pred_event_type,
                                        true_type=event["type"],
                                        input_template=query,
                                        trigger_left=trigger_left,
                                        trigger_right=trigger_right,
                                        argu_left=-1,
                                        argu_right=-1
                                    )
                                    self.examples.append(example)
                            else:
                                # golden label
                                key = str(item["id"]) + "_" + trigger["id"]
                                arguments_per_trigger = dict(id=key, role=role, arguments=[])
                                arguments_per_trigger["pred_type"] = pred_event_type
                                arguments_per_trigger["true_type"] = event["type"]
                                for argument in trigger["arguments"]:
                                    if argument["role"] == role:
                                        arguments_per_role = {
                                            "role": role,
                                            "mentions": []
                                        }
                                        for mention in argument["mentions"]:
                                            left_pos = len(item["text"][:mention["position"][0]].split())
                                            right_pos = len(item["text"][:mention["position"][1]].split())
                                            arguments_per_role["mentions"].append({
                                                "position": [left_pos, right_pos-1]
                                            })
                                        arguments_per_trigger["arguments"].append(arguments_per_role)
                                self.data_for_evaluation["golden_arguments"].append(arguments_per_trigger)
                                # one instance per query 
                                example = EAEInputExample(
                                    example_id=trigger["id"],
                                    text=item["text"],
                                    pred_type=pred_event_type,
                                    true_type=event["type"],
                                    input_template=query,
                                    trigger_left=trigger_left,
                                    trigger_right=trigger_right
                                )
                                self.examples.append(example)
                        trigger_idx += 1
                # negative triggers 
                for neg in item["negative_triggers"]:
                    trigger_idx += 1  

    def word_offset_to_subword_offset_start(self, position, wordids):
        if not position:
            return -1 
        subword_idx = -1
        for i, wordid in enumerate(wordids):
                if position == wordid:
                    subword_idx = i
                    return subword_idx
        return subword_idx
    

    def word_offset_to_subword_offset_end(self, position, wordids):
        if not position:
            return -1 
        subword_idx = -1
        for i, wordid in enumerate(wordids):
                if position == wordid:
                    subword_idx = i
        return subword_idx
        

    def convert_examples_to_features(self):
        self.input_features = []
        self.data_for_evaluation["text_range"] = []
        self.data_for_evaluation["subword_to_word"] = []
        self.data_for_evaluation["text"] = []
        whitespace = True if self.config.language == "English" else False 
        for example in tqdm(self.examples, desc="Processing features for MRC"):
            # context 
            input_context = self.tokenizer(example.text.split(),
                                           truncation=True,
                                           max_length=self.config.max_seq_length,
                                           is_split_into_words=True)
            # template 
            input_template = self.tokenizer(example.input_template.split(), 
                                            truncation=True,
                                            padding="max_length",
                                            max_length=self.config.max_seq_length,
                                            is_split_into_words=True)
            # concatnate 
            input_ids = input_context["input_ids"] + input_template["input_ids"]
            attention_mask = input_context["attention_mask"] + input_template["attention_mask"]
            # truncation
            input_ids = input_ids[:self.config.max_seq_length]
            attention_mask = attention_mask[:self.config.max_seq_length]
            # output labels
            context_wordids = input_context.word_ids()
            start_position = self.word_offset_to_subword_offset_start(example.argu_left, context_wordids)
            end_position = self.word_offset_to_subword_offset_start(example.argu_right, context_wordids)
            start_position = 0 if start_position == -1 else start_position
            end_position = 0 if end_position == -1 else end_position
            # data for evaluation
            text_range = dict()
            text_range["start"] = 1
            text_range["end"] = text_range["start"] + sum(input_context["attention_mask"][1:])
            self.data_for_evaluation["text_range"].append(text_range)
            self.data_for_evaluation["text"].append(example.text)
            self.data_for_evaluation["subword_to_word"].append(context_wordids)
            # features
            features = EAEInputFeatures(
                example_id = example.example_id,
                input_ids = input_ids,
                attention_mask = attention_mask,
                start_positions=start_position,
                end_positions=end_position
            )
            self.input_features.append(features)