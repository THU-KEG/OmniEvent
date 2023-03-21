import os
from typing import List, Optional
from xml.dom.minidom import parse
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import re
import pdb
import random
import json
import numpy as np
import copy
import argparse
from pathlib import Path
from collections import defaultdict


class StanfordCoreNLPv2(StanfordCoreNLP):
    """StanfordCoreNLP toolkit for sentence tokenization.

    StanfordCoreNLP toolkit for sentence tokenization, tokenizing the input sentence into a list of sentences to
    satisfy the sentence-level event extraction.
    """
    def __init__(self, path, port=8088):
        super(StanfordCoreNLPv2, self).__init__(path, port=port)  # add port=8888 and comment line84-85 in corenlp.py on MacOS

    def sent_tokenize(self,
                      sentence: str):
        r_dict = self._request('ssplit,tokenize', sentence)
        tokens = [[token['originalText'] for token in s['tokens']] for s in r_dict['sentences']]
        spans = [[(token['characterOffsetBegin'], token['characterOffsetEnd']) for token in s['tokens']] for s in
                 r_dict['sentences']]
        return tokens, spans


class Extractor():
    """An extractor that extracts events triggers, entities, and negative triggers from the dataset.

    An extractor that extracts event triggers, entities, and negative triggers from the dataset, returning the
    annotations in lists of dictionaries, respectively.

    Attributes:
        dirs (`List[str]`):
            A list of strings indicating the subdirectories of the ACE2005 dataset.
        split_tags (`Dict[str, List[str]]`):
            A dictionary indicating the special tokens within each subdirectory, which are regarded as split tags.
        Events (`List[Dict]`):
            A list of dictionaries representing the event trigger annotations within the dataset.
        None_events (`List[Dict]`):
            A list of dictionaries representing the negative event mention annotations within the dataset.
        Entities (`List[Dict]`):
            A list of dictionaries representing the entity annotations within the dataset.
    """

    def __init__(self,
                 args) -> None:
        """Constructs an `Extractor`."""
        self.dirs = ['bc', 'bn', 'cts', 'nw', 'un', 'wl']
        self.split_tags = {'bc': ["</SPEAKER>", '</TURN>', '<HEADLINE>', '</HEADLINE>'],
                           'bn': ["<TURN>", "</TURN>"],
                           "cts": ["</SPEAKER>", "</TURN>"],
                           'nw': ['<TEXT>', '</TEXT>', '<HEADLINE>', '</HEADLINE>'],
                           'un': ['</SUBJECT>', '<HEADLINE>', '</HEADLINE>', '<SUBJECT>', '</POST>', '<QUOTE'],
                           'wl': ['</POSTDATE>', '</POST>', '<HEADLINE>', '</HEADLINE>', '<TEXT>', '</TEXT>']}
        self.Events = []
        self.None_events = []
        self.Entities = []
        self.args = args

    def find_index(self,
                   offsets: List[List[int]],
                   offset: List[int]):  # offsets [) offset []
        """Finds the actual word-level offset of the mention."""
        idx_start = -1
        idx_end = -1
        for j, _offset in enumerate(offsets):
            if idx_start == -1 and _offset[0] <= offset[0] and _offset[1] > offset[0]:
                idx_start = j
            if idx_end == -1 and _offset[0] <= offset[1] and _offset[1] > offset[1]:
                idx_end = j
                break
        assert idx_start != -1 and idx_end != -1
        return idx_start, idx_end

    def sentence_distillation(self,
                              sents: List[str],
                              offsets,
                              dir: str):
        """Remove the xml elements within the source text."""
        mark_split_tag = self.split_tags[dir]

        new_sents = []
        new_offsets = []

        if dir == 'cst':
            sents = sents[1:]
            offsets = offsets[1:]

        for i, sent in enumerate(sents):
            offset_per_sentence = offsets[i]
            select = True

            start_posi = 0
            for j, token in enumerate(sent):
                if bool(sum([token.startswith(e) for e in mark_split_tag])):
                    subsent = sent[start_posi:j]
                    suboffset = offset_per_sentence[start_posi:j]
                    if select and len(subsent) > 0:
                        assert (0, 0) not in suboffset
                        new_sents.append(subsent)
                        new_offsets.append(suboffset)
                    start_posi = j + 1
                    select = True
                elif token.startswith('<'):
                    select = False

            subsent = sent[start_posi:]
            suboffset = offset_per_sentence[start_posi:]
            if select and len(subsent) > 0:
                assert (0, 0) not in suboffset
                new_sents.append(subsent)
                new_offsets.append(suboffset)
        return new_sents, new_offsets

    def correct_offsets(self,
                        sents: List[str],
                        offsets):
        """Corrects the offsets of sentences after removing xml elements."""
        new_offsets = []
        minus = 0
        for i, offsets_per_sentence in enumerate(offsets):
            sentence = sents[i]
            new_offsets_per_sentence = []
            for j, offset in enumerate(offsets_per_sentence):
                if sentence[j].startswith('<'):
                    new_offsets_per_sentence.append((0, 0))
                    minus += len(sentence[j])

                else:
                    new_offsets_per_sentence.append((offset[0] - minus, offset[1] - minus))
            new_offsets.append(new_offsets_per_sentence)
        return sents, new_offsets

    def Files_Extract(self) -> None:
        """Extracts the filenames containing events, source texts, and amps."""
        self.event_files = {}
        self.source_files = {}
        self.amp_files = []
        for dir in self.dirs:
            path = self.args.ACE_FILES + '/' + dir + '/timex2norm'
            files = os.listdir(path)
            self.event_files[dir] = [file for file in files if file.endswith('.apf.xml')]
            self.source_files[dir] = [file for file in files if file.endswith('.sgm')]
            for file in self.source_files[dir]:
                with open(path + '/' + file, 'r') as f:
                    text = f.read()
                if '&amp;' in text:
                    self.amp_files.append(file[:-3])

        srclen = 0
        evtlen = 0
        for dir in self.dirs:
            srclen += len(self.source_files[dir])
            evtlen += len(self.event_files[dir])
        assert evtlen == srclen
        assert evtlen == 599

    def Entity_Extract(self) -> None:
        """Extracts the entity annotations from the dataset."""
        all_ents = 0
        for dir in self.dirs:
            path = self.args.ACE_FILES + '/' + dir + '/timex2norm'
            files = self.event_files[dir]
            for file in files:
                DOMtree = parse(path + "/" + file)
                collection = DOMtree.documentElement
                all_ents += len(collection.getElementsByTagName("entity"))
                tags = ["entity", "value", "timex2"]
                for tag in tags:
                    mention_tag = f"{tag}_mention"
                    elements = collection.getElementsByTagName(tag)
                    for element in elements:
                        mention = element.getElementsByTagName(mention_tag)
                        ent_id = element.getAttribute("ID")
                        ent_type = element.getAttribute("SUBTYPE")
                        for sample in mention:
                            mention_id = sample.getAttribute("ID")
                            start = int(sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[
                                            0].getAttribute("START"))
                            end = int(sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[
                                          0].getAttribute("END"))
                            name = str(
                                sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].childNodes[
                                    0].data)
                            entity_info = (name, start, end, file, dir, ent_id, mention_id, ent_type)
                            self.Entities.append(entity_info)
        self.Entities = list(set(self.Entities))
        self.Entities = [{'entity_id': e[5], "mention_id": e[6], "type": e[7], 'name': e[0], 'start': e[1], 'end': e[2],
                          'file': e[3], 'dir': e[4], 'role': 'None'} for e in self.Entities]
        print("Total %d mentions, %d entities." % (len(self.Entities), all_ents))

    def Event_Extract(self) -> None:
        """Extracts the event annotations from the dataset."""
        nlp = StanfordCoreNLPv2(self.args.corenlp_path)
        offsets2idx = {}
        for dir in self.dirs:
            path = self.args.ACE_FILES + '/' + dir + '/timex2norm'
            files = self.event_files[dir]
            for file in files:
                DOMtree = parse(path + "/" + file)
                collection = DOMtree.documentElement
                events = collection.getElementsByTagName("event")
                Entities = [e for e in self.Entities if e['dir'] == dir and e['file'] == file]
                for event in events:
                    event_id = event.getAttribute("ID")
                    event_type = str(event.getAttribute("TYPE")) + "." + str(event.getAttribute("SUBTYPE"))
                    event_mentions = event.getElementsByTagName("event_mention")
                    for event_mention in event_mentions:
                        mention_id = event_mention.getAttribute("ID")
                        event_info = event_mention.getElementsByTagName("ldc_scope")[0].getElementsByTagName("charseq")[
                            0]
                        sent = str(event_info.childNodes[0].data)
                        start = int(event_info.getAttribute("START"))
                        end = int(event_info.getAttribute("END"))

                        trigger_info = event_mention.getElementsByTagName("anchor")[0].getElementsByTagName("charseq")[
                            0]
                        trigger = str(trigger_info.childNodes[0].data)
                        trigger_start = int(trigger_info.getAttribute("START"))
                        trigger_end = int(trigger_info.getAttribute("END"))

                        entities = [copy.deepcopy(e) for e in Entities if e['start'] >= start and e['end'] <= end]

                        map_entity = {(e['start'], e['end']): i for i, e in enumerate(entities)}

                        arguments = event_mention.getElementsByTagName("event_mention_argument")
                        for argument in arguments:
                            role = str(argument.getAttribute("ROLE"))
                            argument_info = argument.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[
                                0]
                            argument_name = str(argument_info.childNodes[0].data)
                            argument_start = int(argument_info.getAttribute("START"))
                            argument_end = int(argument_info.getAttribute("END"))
                            assert (argument_start, argument_end) in map_entity
                            entity_id = map_entity[(argument_start, argument_end)]
                            assert argument_name == entities[entity_id]['name']
                            entities[entity_id]['role'] = role
                        tokens, offsets = nlp.word_tokenize(sent, True)

                        plus = 0
                        for j, token in enumerate(tokens):
                            st = offsets[j][0] + plus
                            if file[:-7] in self.amp_files:
                                plus += 4 * token.count('&')
                            ed = offsets[j][1] + plus
                            offsets[j] = (st, ed)

                        tokens_offsets = [(e[0] + start, e[1] - 1 + start) for e in offsets]
                        find_offsets = [(e[0] + start, e[1] + start) for e in offsets]
                        trigger_s, trigger_e = self.find_index(find_offsets, (trigger_start, trigger_end))
                        trigger_offsets = tokens_offsets[trigger_s:trigger_e + 1]
                        trigger_tokens = tokens[trigger_s:trigger_e + 1]
                        _entities = []
                        for e in entities:
                            idx_start, idx_end = self.find_index(find_offsets, (e['start'], e['end']))
                            entity_tokens = tokens[idx_start:idx_end + 1]
                            entity_offsets = tokens_offsets[idx_start:idx_end + 1]
                            entity_start = entity_offsets[0][0]
                            entity_end = entity_offsets[-1][1]
                            entity_info = {'tokens': entity_tokens,
                                           'offsets': entity_offsets,
                                           'start': entity_start,
                                           'end': entity_end,
                                           'idx_start': idx_start,
                                           'idx_end': idx_end,
                                           'role': e['role'],
                                           "entity_id": e["entity_id"],
                                           "mention_id": e["mention_id"],
                                           "type": e["type"]
                                           }
                            _entities.append(entity_info)
                        event_summary = {
                            "event_id": event_id,
                            "trigger_id": mention_id,
                            "tokens": tokens,
                            'offsets': tokens_offsets,
                            "event_type": event_type,
                            "start": start,
                            "end": end,
                            "trigger_tokens": trigger_tokens,
                            "trigger_start": trigger_s,
                            "trigger_end": trigger_e,
                            'trigger_offsets': trigger_offsets,
                            "entities": _entities,
                            'file': file[:-8],
                            'dir': dir
                        }
                        offsets_join = str(event_summary['start']) + '_' + str(event_summary['end']) + "_" + \
                                       event_summary['file'] + "_" + event_summary['dir']
                        event_summary["offset_join"] = offsets_join
                        self.Events.append(event_summary)
        nlp.close()

    def None_event_Extract(self) -> None:
        """Extract negative event mentions from the dataset."""
        nlp = StanfordCoreNLPv2(self.args.corenlp_path, port=8095)
        for dir in self.dirs:
            path = self.args.ACE_FILES + '/' + dir + '/timex2norm'
            files = self.source_files[dir]
            for file in files:
                event_in_this_file = [(e['start'], e['end']) for e in self.Events if
                                      e['file'] == file[:-4] and e['dir'] == dir]
                Entities = [e for e in self.Entities if
                            e['dir'] == dir and e['file'][:-7] == file[:-3]]
                with open(path + '/' + file, 'r') as f:
                    text = f.read()
                sents, offsets = nlp.sent_tokenize(text)
                sents, offsets = self.correct_offsets(sents, offsets)
                sents, offsets = self.sentence_distillation(sents, offsets, dir)

                new_sents = []
                new_offsets = []
                for j, sent in enumerate(sents):
                    offset = offsets[j]
                    select = True
                    for event in event_in_this_file:
                        if (offset[0][0] >= event[0] and offset[0][0] <= event[1]) or \
                                (offset[-1][1] - 1 >= event[0] and offset[-1][1] - 1 <= event[1]):
                            select = False
                            break
                    if select:
                        new_sents.append(sent)
                        new_offsets.append(offset)

                sents = new_sents
                offsets = new_offsets

                for i, sent in enumerate(sents):
                    offset = offsets[i]
                    tokens = sent
                    start = offset[0][0]
                    end = offset[-1][1] - 1
                    tokens_offset = [(e[0], e[1] - 1) for e in offset]
                    event_type = 'None'
                    trigger_tokens = []
                    trigger_offsets = []
                    trigger_start = -1
                    trigger_end = -1
                    entities = []

                    _entities = [copy.deepcopy(e) for e in Entities if e["start"] >= start and e["end"] <= end]
                    for e in _entities:
                        try:
                            idx_start, idx_end = self.find_index(offset, (e["start"], e["end"]))
                        except:
                            print("An entity can't be found.", e)
                            continue
                        entity_info = {'tokens': sent[idx_start:idx_end + 1],
                                       'role': 'None',
                                       'offsets': [(e[0], e[1] - 1) for e in offset[idx_start:idx_end + 1]],
                                       'start': offset[idx_start][0],
                                       'end': offset[idx_end][1] - 1,
                                       'idx_start': idx_start,
                                       'idx_end': idx_end,
                                       "entity_id": e["entity_id"],
                                       "mention_id": e["mention_id"],
                                       "type": e["type"]}
                        entities.append(entity_info)
                    none_event_summary = {
                        'tokens': tokens,
                        'start': start,
                        'end': end,
                        'offsets': tokens_offset,
                        'event_type': event_type,
                        'trigger_tokens': trigger_tokens,
                        'trigger_start': trigger_start,
                        'trigger_end': trigger_end,
                        'trigger_offsets': trigger_offsets,
                        'entities': entities,
                        'file': file[:-4],
                        'dir': dir
                    }
                    self.None_events.append(none_event_summary)
        nlp.close()

    def process(self) -> None:
        """Converts the word-level position annotations of event triggers, entities, and none trigger mentions into
           character-level position annotations."""
        Events = []
        # convert to sentence level
        events_in_sens = defaultdict(list)
        for event in self.Events:
            events_in_sens[event["offset_join"]].append(event)
        # loop for recording events in sentence
        for sen_id, events in events_in_sens.items():
            event_to_triggers = defaultdict(list)
            for event in events:
                event_to_triggers[event["event_id"]].append(event)
            tokens = events[0]["tokens"]
            trigger_position = {i: False for i in range(len(tokens))}
            refined_sen_events = {}
            refined_sen_events["id"] = sen_id
            refined_sen_events["text"] = " ".join(tokens)
            refined_sen_events["file"] = events[0]["file"]
            refined_sen_events["events"] = []
            for event_id, triggers in event_to_triggers.items():
                refined_event = {}
                refined_event["type"] = triggers[0]["event_type"]
                refined_event["triggers"] = []
                for trigger in triggers:  # trigger level
                    for pos in range(trigger["trigger_start"], trigger["trigger_end"] + 1):
                        trigger_position[pos] = True
                    refined_trigger = {}
                    refined_trigger["id"] = trigger["trigger_id"]
                    refined_trigger["trigger_word"] = " ".join(trigger["trigger_tokens"])
                    refined_trigger["position"] = token_pos_to_char_pos(trigger["tokens"], [trigger["trigger_start"],
                                                                                            trigger["trigger_end"] + 1])
                    refined_trigger["arguments"] = []
                    arguments = defaultdict(list)
                    for entity in trigger["entities"]:
                        if entity["role"] == "None":
                            continue
                        argu = dict()
                        argu["mention_id"] = entity["mention_id"]
                        argu["mention"] = " ".join(entity["tokens"])
                        argu["position"] = token_pos_to_char_pos(tokens, [entity["idx_start"], entity["idx_end"] + 1])
                        arguments["{}[SEP]{}".format(entity["entity_id"], entity["role"])].append(argu)
                    for key, mentions in arguments.items():
                        ent_id, role = key.split("[SEP]")
                        refined_role = dict(id=ent_id, role=role)
                        refined_role["mentions"] = mentions
                        refined_trigger["arguments"].append(refined_role)
                    refined_event["triggers"].append(refined_trigger)
                refined_sen_events["events"].append(refined_event)
            # negative triggers
            refined_sen_events["negative_triggers"] = []
            for i in range(len(tokens)):
                if trigger_position[i]:
                    continue
                if tokens[i] == "":
                    continue
                _event = {
                    "id": len(refined_sen_events["negative_triggers"]),
                    "trigger_word": tokens[i],
                    "position": token_pos_to_char_pos(tokens, [i, i + 1])
                }
                refined_sen_events["negative_triggers"].append(_event)
            # process all entities
            refined_sen_events["entities"] = []
            id2mentions = defaultdict(list)
            for entity in events[0]["entities"]:
                id2mentions[entity["entity_id"]].append(entity)
            for id, mentions in id2mentions.items():
                argu = dict(id=id, type=mentions[0]["type"], mentions=[])
                for mention in mentions:
                    argu["mentions"].append({
                        "mention_id": mention["mention_id"],
                        "mention": " ".join(mention["tokens"]),
                        "position": token_pos_to_char_pos(tokens, [mention["idx_start"], mention["idx_end"] + 1])
                    })
                refined_sen_events["entities"].append(argu)
            Events.append(refined_sen_events)

        # process none events
        for none_event in self.None_events:
            refined_sen_events = dict(id=len(Events))
            refined_sen_events["text"] = " ".join(none_event['tokens'])
            refined_sen_events["events"] = []
            refined_sen_events["negative_triggers"] = []
            refined_sen_events["file"] = none_event["file"]
            for i in range(len(none_event['tokens'])):
                if none_event['tokens'][i] == "":
                    continue
                _none_event = {
                    "id": len(refined_sen_events["negative_triggers"]),
                    'trigger_word': none_event['tokens'][i],
                    'position': token_pos_to_char_pos(none_event["tokens"], [i, i + 1])
                }
                refined_sen_events["negative_triggers"].append(_none_event)
            # entities
            refined_sen_events["entities"] = []
            id2mentions = defaultdict(list)
            for entity in none_event["entities"]:
                id2mentions[entity["entity_id"]].append(entity)
            for id, mentions in id2mentions.items():
                argu = dict(id=id, type=mentions[0]["type"], mentions=[])
                for mention in mentions:
                    argu["mentions"].append({
                        "mention_id": mention["mention_id"],
                        "mention": " ".join(mention["tokens"]),
                        "position": token_pos_to_char_pos(none_event["tokens"], [mention["idx_start"], mention["idx_end"] + 1])
                    })
                refined_sen_events["entities"].append(argu)
            Events.append(refined_sen_events)
        # record
        self.Events = Events

    def Extract(self) -> None:
        """Extracts the entities, events and negative mentions, splits the training, validation, and testing set,
           and writes the datasets into json files."""
        if os.path.exists(os.path.join(self.args.ACE_DUMP, 'train.json')):
            print('--Already Exists Files--')
            return

        self.Files_Extract()
        print('--File Extraction Finish--')
        self.Entity_Extract()
        print('--Entity Extraction Finish--')
        self.Event_Extract()
        print('--Event Mention Extraction Finish--')
        self.None_event_Extract()
        print('--Negetive Mention Extraction Finish--')
        self.process()
        print('--Preprocess Data Finish--')

        # Use fix split
        splits = {'train': [], 'dev': [], 'test': []}

        splits_name = ['train', 'dev', 'test']
        for split in splits_name:
            with open(os.path.join(self.args.ACE_SPLITS, '{}.txt'.format(split)), 'r') as f:
                split_file = f.readline().strip()
                while split_file:
                    splits[split].append(split_file)
                    split_file = f.readline().strip()

        test_files = splits['test']
        dev_files = splits['dev']
        train_files = splits['train']
        test_set = [instance for instance in self.Events if
                    instance['file'].replace('.', '_').replace('-', '_') in test_files]
        dev_set = [instance for instance in self.Events if
                   instance['file'].replace('.', '_').replace('-', '_') in dev_files]
        train_set = [instance for instance in self.Events if
                     instance['file'].replace('.', '_').replace('-', '_') in train_files]

        with open(os.path.join(self.args.ACE_DUMP, 'train.json'), 'w') as f:
            json.dump(train_set, f, indent=4)
        with open(os.path.join(self.args.ACE_DUMP, 'valid.json'), 'w') as f:
            json.dump(dev_set, f, indent=4)
        with open(os.path.join(self.args.ACE_DUMP, 'test.json'), 'w') as f:
            json.dump(test_set, f, indent=4)


def token_pos_to_char_pos(tokens: List[str],
                          token_pos: List[int]) -> List[int]:
    """Converts the token-level position of a mention into character-level.

    Converts the token-level position of a mention into character-level by counting the number of characters before the
    start position of the mention. The end position could then be derived by adding the character-level start position
    and the length of the mention's span.

    Args:
        tokens (`List[str]`):
            A list of strings representing the tokens within the source text.
        token_pos (`List[int]`):
            A list of integers indicating the word-level start and end position of the mention.

    Returns:
        `List[int]`:
            A list of integers representing the character-level start and end position of the mention.
    """
    word_span = " ".join(tokens[token_pos[0]:token_pos[1]])
    char_start, char_end = -1, -1
    curr_pos = 0
    for i, token in enumerate(tokens):
        if i == token_pos[0]:
            char_start = curr_pos
            break
        curr_pos += len(token) + 1
    assert char_start != -1
    char_end = char_start + len(word_span)
    sen = " ".join(tokens)
    assert sen[char_start:char_end] == word_span
    return [char_start, char_end]


def convert_ace2005_to_unified(output_dir: str,
                               file_name: str,
                               dump: Optional[bool] = True) -> None:
    """Convert ACE2005 dataset to the unified format.

    Extract the information from the original ACE2005 dataset and convert the format to a unified OpenEE dataset. The
    converted dataset is written to a json file.

    Args:
        output_dir (`str`):
            A string indicating the output directory of the output file.
        file_name (`str`):
            A string indicating the filename of the output file.
        dump (`bool`, `optional`, defaults to `True`):
            A boolean variable indicating whether or not to write the manipulated dataset into a json file.
    """
    data = json.load(open(os.path.join(output_dir, file_name)))
    label2id = dict(NA=0)
    role2id = dict(NA=0)
    print("We got %d instances" % len(data))
    with open(os.path.join(output_dir, file_name.replace(".json", ".unified.jsonl")), "w") as f:
        for instance in data:
            for event in instance["events"]:
                if event["type"] not in label2id:
                    label2id[event["type"]] = len(label2id)
                for trigger in event["triggers"]:
                    for argument in trigger["arguments"]:
                        if argument["role"] not in role2id:
                            role2id[argument["role"]] = len(role2id)
            del instance["file"]
            f.write(json.dumps(instance) + "\n")
    if "train" in file_name:
        json.dump(label2id, open(os.path.join(output_dir, "label2id.json"), "w"))
        json.dump(role2id, open(os.path.join(output_dir, "role2id.json"), "w"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ACE2005")
    arg_parser.add_argument("--data_dir", type=str, default=None)
    arg_parser.add_argument("--ACE_SPLITS", type=str, default=None)
    arg_parser.add_argument("--ACE_FILES", type=str, default=None)
    arg_parser.add_argument("--ACE_DUMP", type=str, default=None)
    arg_parser.add_argument("--corenlp_path", type=str, default=None)
    args = arg_parser.parse_args()
    args.ACE_FILES = os.path.join(args.data_dir, "data/English")

    dump_path = Path(args.ACE_DUMP)
    dump_path.mkdir(parents=True, exist_ok=True)

    # process original files
    extractor = Extractor(args)
    extractor.Extract()

    # convert formats
    convert_ace2005_to_unified(args.ACE_DUMP, "train.json")
    convert_ace2005_to_unified(args.ACE_DUMP, "valid.json")
    convert_ace2005_to_unified(args.ACE_DUMP, "test.json")
