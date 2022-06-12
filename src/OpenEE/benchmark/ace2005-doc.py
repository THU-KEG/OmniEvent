import os
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
    def __init__(self,path):
        super(StanfordCoreNLPv2,self).__init__(path)
    def sent_tokenize(self,sentence):
        r_dict = self._request('ssplit,tokenize', sentence)
        tokens = [[token['originalText'] for token in s['tokens']] for s in r_dict['sentences']]
        spans = [[(token['characterOffsetBegin'], token['characterOffsetEnd']) for token in s['tokens']] for s in r_dict['sentences'] ]
        return tokens, spans

class Extractor():
    def __init__(self, args):
        self.dirs = ['bc','bn','cts','nw','un','wl']
        self.split_tags ={'bc':["</SPEAKER>",'</TURN>','<HEADLINE>','</HEADLINE>'],
                          'bn':["<TURN>","</TURN>"],
                          "cts":["</SPEAKER>","</TURN>"],
                          'nw':['<TEXT>','</TEXT>','<HEADLINE>','</HEADLINE>'],
                          'un':['</SUBJECT>','<HEADLINE>','</HEADLINE>','<SUBJECT>','</POST>','<QUOTE'],
                          'wl':['</POSTDATE>','</POST>','<HEADLINE>','</HEADLINE>','<TEXT>','</TEXT>']}
        self.Events = []
        self.None_events = []
        self.Entities = []
        self.args = args 

    def find_index(self,offsets,offset): #offsets [) offset []
        idx_start = -1
        idx_end = -1
        for j, _offset in enumerate(offsets):
            if idx_start == -1 and _offset[0] <= offset[0] and _offset[1] > offset[0]:
                idx_start = j
            if idx_end == -1 and _offset[0] <= offset[1] and _offset[1] > offset[1]:
                idx_end = j
                break
        assert idx_start!=-1 and idx_end!=-1
        return idx_start,idx_end

    def sentence_distillation(self,sents,offsets,dir):
        mark_split_tag = self.split_tags[dir]

        new_sents = []
        new_offsets = []

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
        return new_sents,new_offsets


    def correct_offsets(self,sents,offsets):
        new_offsets = []
        minus = 0
        for i,offsets_per_sentence in enumerate(offsets):
            sentence = sents[i]
            new_offsets_per_sentence = []
            for j,offset in enumerate(offsets_per_sentence):
                if sentence[j].startswith('<'):
                    new_offsets_per_sentence.append((0,0))
                    minus+=len(sentence[j])

                else:
                    new_offsets_per_sentence.append((offset[0]-minus,offset[1]-minus))
            new_offsets.append(new_offsets_per_sentence)
        return sents,new_offsets


    def Files_Extract(self):
        self.event_files = {}
        self.source_files = {}
        self.amp_files = []
        for dir in self.dirs:
            path = self.args.ACE_FILES+'/'+dir+'/timex2norm'
            files = os.listdir(path)
            self.event_files[dir] = [file for file in files if file.endswith('.apf.xml')]
            self.source_files[dir] = [file for file in files if file.endswith('.sgm')]
            for file in self.source_files[dir]:
                with open(path+'/'+file,'r') as f:
                    text = f.read()
                if '&amp;' in text:
                    self.amp_files.append(file[:-3])

        srclen = 0
        evtlen = 0
        for dir in self.dirs:
            srclen+=len(self.source_files[dir])
            evtlen+=len(self.event_files[dir])
        assert evtlen==srclen
        assert evtlen==599


    def Entity_Extract(self):
        all_ents = 0
        for dir in self.dirs:
            path = self.args.ACE_FILES+'/'+dir+'/timex2norm'
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
                            start = int(sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].getAttribute("START"))
                            end = int(sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].getAttribute("END"))
                            name = str(sample.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0].childNodes[0].data)
                            entity_info = (name,start,end,file,dir,ent_id,ent_type)
                            self.Entities.append(entity_info)
        self.Entities = list(set(self.Entities))
        self.Entities = [{'id':e[5], "type":e[6], 'name':e[0],'start':e[1],'end':e[2],'file':e[3],'dir':e[4],'role':'None'} for e in self.Entities]
        print("Total %d mentions, %d entities." % (len(self.Entities), all_ents))


    def Event_Extract(self):
        nlp = StanfordCoreNLPv2(self.args.corenlp_path)
        offsets2idx = {}
        for dir in self.dirs:
            path = self.args.ACE_FILES+'/'+dir+'/timex2norm'
            files =self.event_files[dir]
            for file in files:
                DOMtree = parse(path + "/" + file)
                collection = DOMtree.documentElement
                events = collection.getElementsByTagName("event")
                Entities = [e for e in self.Entities if e['dir']==dir and e['file']==file]
                for event in events:
                    event_id = event.getAttribute("ID")
                    event_type =  str(event.getAttribute("SUBTYPE"))
                    event_mentions = event.getElementsByTagName("event_mention")
                    for event_mention in event_mentions:
                        event_info = event_mention.getElementsByTagName("ldc_scope")[0].getElementsByTagName("charseq")[0]
                        sent = str(event_info.childNodes[0].data)
                        start = int(event_info.getAttribute("START"))
                        end = int(event_info.getAttribute("END"))

                        trigger_info = event_mention.getElementsByTagName("anchor")[0].getElementsByTagName("charseq")[0]
                        trigger =  str(trigger_info.childNodes[0].data)
                        trigger_start = int(trigger_info.getAttribute("START"))
                        trigger_end = int(trigger_info.getAttribute("END"))

                        entities = [copy.deepcopy(e) for e in Entities if e['start']>=start and e['end']<=end]

                        map_entity = {(e['start'],e['end']):i for i,e in enumerate(entities)}

                        arguments = event_mention.getElementsByTagName("event_mention_argument")
                        for argument in arguments:
                            role = str(argument.getAttribute("ROLE"))
                            argument_info = argument.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0]
                            argument_name = str(argument_info.childNodes[0].data)
                            argument_start = int(argument_info.getAttribute("START"))
                            argument_end = int(argument_info.getAttribute("END"))
                            assert (argument_start,argument_end) in map_entity
                            entity_id = map_entity[(argument_start,argument_end)]
                            assert argument_name==entities[entity_id]['name']
                            entities[entity_id]['role'] = role
                        tokens,offsets = nlp.word_tokenize(sent,True)

                        plus = 0
                        for j,token in enumerate(tokens):
                            st = offsets[j][0] + plus
                            if file[:-7] in self.amp_files:
                                plus += 4*token.count('&')
                            ed = offsets[j][1] + plus
                            offsets[j] = (st,ed)

                        tokens_offsets = [(e[0] + start, e[1] - 1 + start) for e in offsets]
                        find_offsets = [(e[0]+start,e[1]+start) for e in offsets]
                        trigger_s,trigger_e = self.find_index(find_offsets,(trigger_start,trigger_end))
                        trigger_offsets = tokens_offsets[trigger_s:trigger_e+1]
                        trigger_tokens = tokens[trigger_s:trigger_e+1]
                        _entities = []
                        for e in entities:
                            idx_start,idx_end = self.find_index(find_offsets,(e['start'],e['end']))
                            entity_tokens = tokens[idx_start:idx_end+1]
                            entity_offsets = tokens_offsets[idx_start:idx_end+1]
                            entity_start = entity_offsets[0][0]
                            entity_end = entity_offsets[-1][1]
                            entity_info = {'tokens':entity_tokens,
                                           'offsets':entity_offsets,
                                           'start':entity_start,
                                           'end':entity_end,
                                           'idx_start':idx_start,
                                           'idx_end':idx_end,
                                           'role':e['role'],
                                           "id":e["id"],
                                           "type":e["type"]
                                           }
                            _entities.append(entity_info)
                        event_summary = {"id": event_id,
                                        "tokens": tokens,
                                         'offsets':tokens_offsets,
                                         "event_type": event_type,
                                         "start": start,
                                         "end": end,
                                         "trigger_tokens": trigger_tokens,
                                         "trigger_start": trigger_s,
                                         "trigger_end": trigger_e,
                                         'trigger_offsets': trigger_offsets,
                                         "entities": _entities,
                                         'file': file[:-8],
                                         'dir': dir}
                        self.Events.append(event_summary)
        nlp.close()
    

    def reindex(self, old_idx_to_new_idx, old_idx, end=False):
        if end:
            return old_idx_to_new_idx[old_idx-1]+1
        return old_idx_to_new_idx[old_idx]


    def doc_event_Extract(self):
        nlp = StanfordCoreNLPv2(self.args.corenlp_path)
        all_events = []
        for dir in self.dirs:
            path = self.args.ACE_FILES+'/'+dir+'/timex2norm'
            files =self.source_files[dir]
            for file in tqdm(files):
                event_in_this_file = []
                for event in self.Events:
                    if event["file"] == file[:-4] and event["dir"] == dir:
                        event_in_this_file.append(event)
                Entities = [e for e in self.Entities if e['dir']==dir and e['file'][:-7]==file[:-3]]
                with open(path+'/'+file,'r') as f:
                    text = f.read()
                sents,offsets = nlp.sent_tokenize(text)
                sents,offsets = self.correct_offsets(sents,offsets)
                sents,offsets = self.sentence_distillation(sents,offsets,dir)

                doc = ""
                old_idx_to_new_idx = {}
                current_length = 0
                for token_offsets, sent in zip(offsets, sents):
                    for token_offset, token in zip(token_offsets, sent):
                        for _offset in range(token_offset[0], token_offset[1]):
                            old_idx_to_new_idx[_offset] = current_length
                            current_length += 1
                        doc += token 
                        doc += " "
                        current_length += 1
                        assert current_length == len(doc)
                doc = doc[:-1]
                tokens = doc.split()

                event_id_to_mentions = defaultdict(list)
                for event in event_in_this_file:
                    event_id_to_mentions[event["id"]].append(event)
                entity_id_to_mentions = defaultdict(list)
                for entity in Entities:
                    entity_id_to_mentions[entity["id"]].append(entity)

                per_event = {
                    "id": file,
                    "sentence": doc,
                    "events": [],
                    "negative_triggers": [],
                    "entities": []
                }
                trigger_starts = []
                for event_id, event_mentions in event_id_to_mentions.items():
                    event = {}
                    event["type"] = event_mentions[0]["event_type"]                   
                    event["triggers"] = []
                    event["arguments"] = []
                    arguments = defaultdict(list)
                    for trigger_id, trigger in enumerate(event_mentions):
                        char_pos = [trigger["trigger_offsets"][0][0], trigger["trigger_offsets"][-1][1]+1]
                        position = [self.reindex(old_idx_to_new_idx, char_pos[0]), 
                                    self.reindex(old_idx_to_new_idx, char_pos[1], True)]
                        trigger_word = doc[position[0]:position[1]]
                        event["triggers"].append({
                            "id": f"{event_id}-{trigger_id}",
                            "trigger_word": trigger_word,
                            "position": position
                        })
                        trigger_starts.append(position[0])
                        for entity in trigger["entities"]:
                            if entity["role"] == "None":
                                continue
                            argu = dict()
                            char_pos = [entity["offsets"][0][0], entity["offsets"][-1][1]+1]
                            argu["position"] = [self.reindex(old_idx_to_new_idx, char_pos[0]), 
                                    self.reindex(old_idx_to_new_idx, char_pos[1], True)]
                            argu["mention"] = doc[argu["position"][0]:argu["position"][1]]
                            arguments["{}[SEP]{}".format(entity["id"], entity["role"])].append(argu)
                    for key, mentions in arguments.items():
                        ent_id, role = key.split("[SEP]")
                        refined_role = dict(id=ent_id, role=role)
                        refined_role["mentions"] = mentions 
                        event["arguments"].append(refined_role)
                    # append 
                    per_event["events"].append(event)

                # negative triggers
                curr_pos = 0
                for i in range(len(tokens)):
                    if curr_pos in trigger_starts or tokens[i] == "":
                        curr_pos += len(tokens[i]) + 1
                        continue
                    curr_pos += len(tokens[i]) + 1
                    _event = {
                            "id": len(per_event["negative_triggers"]),
                            "trigger_word": tokens[i],
                            "position": token_pos_to_char_pos(tokens, [i, i+1])
                    }
                    assert _event["trigger_word"] == doc[_event["position"][0]:_event["position"][1]]
                    per_event["negative_triggers"].append(_event)
                # entities 
                for id, mentions in entity_id_to_mentions.items():
                    argu = dict(id=id, type=mentions[0]["type"], mentions=[])
                    for mention in mentions:
                        char_pos = [mention["start"], mention["end"]+1]
                        try:
                            per_mention = {
                                "position": [self.reindex(old_idx_to_new_idx, char_pos[0]), 
                                            self.reindex(old_idx_to_new_idx, char_pos[1], True)]
                            }
                        except:
                            # print("An entity is dismissed.", mention)
                            continue 
                        per_mention["mention"] = doc[per_mention["position"][0]:per_mention["position"][1]]
                        argu["mentions"].append(per_mention)
                    per_event["entities"].append(argu)
                # append 
                per_event["file"] = file[:-4]
                all_events.append(per_event)
        self.Events = all_events
        nlp.close()


    def Extract(self):
        if os.path.exists(os.path.join(self.args.ACE_DUMP, 'train.json')):
            print('--Already Exists Files--')
            return

        self.Files_Extract()
        print('--File Extraction Finish--')
        self.Entity_Extract()
        print('--Entity Extraction Finish--')
        self.Event_Extract()
        print('--Event Mention Extraction Finish--')
        self.doc_event_Extract()
        print('--Doc Mention Extraction Finish--')

        # Random Split
            # nw = self.source_files['nw']
            # random.shuffle(nw)
            # random.shuffle(nw)
            # other_files = [file for dir in self.dirs for file in self.source_files[dir] if dir!='nw']+nw[40:]
            # random.shuffle(other_files)
            # random.shuffle(other_files)

            # test_files = nw[:40]
            # dev_files = other_files[:30]
            # train_files = other_files[30:]

            # test_set = [instance for instance in self.Events if instance['file']+'.sgm' in test_files]+[instance for instance in self.None_events if instance['file']+".sgm" in test_files]
            # dev_set = [instance for instance in self.Events if instance['file']+'.sgm' in dev_files]+[instance for instance in self.None_events if instance['file']+".sgm" in dev_files]
            # train_set = [instance for instance in self.Events if instance['file']+'.sgm' in train_files]+[instance for instance in self.None_events if instance['file']+".sgm" in train_files]

        # Use fix split
        splits = {'train':[],'dev':[],'test':[]}

        splits_name = ['train','dev','test']
        for split in splits_name:
            with open(os.path.join(self.args.ACE_SPLITS, '{}.txt'.format(split)),'r') as f:
                split_file = f.readline().strip()
                while split_file:
                    splits[split].append(split_file)
                    split_file = f.readline().strip()
        
        test_files = splits['test']
        dev_files = splits['dev']
        train_files = splits['train']
        test_set = [instance for instance in self.Events if instance['file'].replace('.','_').replace('-','_') in test_files]
        dev_set = [instance for instance in self.Events if instance['file'].replace('.','_').replace('-','_') in dev_files]
        train_set = [instance for instance in self.Events if instance['file'].replace('.','_').replace('-','_') in train_files]
            
        with open(os.path.join(self.args.ACE_DUMP, 'train.json'),'w') as f:
            json.dump(train_set, f, indent=4)
        with open(os.path.join(self.args.ACE_DUMP, 'valid.json'),'w') as f:
            json.dump(dev_set, f, indent=4)
        with open(os.path.join(self.args.ACE_DUMP, 'test.json'),'w') as f:
            json.dump(test_set, f, indent=4)
        

def token_pos_to_char_pos(tokens, token_pos):
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


def convert_ace2005_to_unified(output_dir: str, file_name: str, dump=True) -> dict:
    data = json.load(open(os.path.join(output_dir, file_name)))
    label2id = dict(NA=0)    
    role2id = dict(NA=0)
    print("We got %d instances" % len(data))
    with open(os.path.join(output_dir, file_name.replace(".json", ".unified.jsonl")), "w") as f:
        for instance in data:
            for event in instance["events"]:
                if event["type"] not in label2id:
                    label2id[event["type"]] = len(label2id)
                for argument in event["arguments"]:
                    if argument["role"] not in role2id:
                        role2id[argument["role"]] = len(role2id)
            del instance["file"]
            f.write(json.dumps(instance)+"\n")
    if "train" in file_name:
        json.dump(label2id, open(os.path.join(output_dir, "label2id.json"), "w"))
        json.dump(role2id, open(os.path.join(output_dir, "role2id.json"), "w"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ACE2005")
    arg_parser.add_argument("--data_dir", type=str, default="../../../data/ace_2005_td_v7")
    arg_parser.add_argument("--ACE_SPLITS", type=str, default=None)
    arg_parser.add_argument("--ACE_FILES", type=str, default=None)
    arg_parser.add_argument("--ACE_DUMP", type=str, default=None)
    arg_parser.add_argument("--corenlp_path", type=str, default=None)
    args = arg_parser.parse_args()
    args.ACE_SPLITS = os.path.join(args.data_dir, "splits")
    args.ACE_FILES = os.path.join(args.data_dir, "data/English")
    dump_path = Path(os.path.join(args.data_dir, "English-doc"))
    dump_path.mkdir(exist_ok=True)
    args.ACE_DUMP = dump_path

    # process original files 
    extractor = Extractor(args)
    extractor.Extract()

    # convert formats
    convert_ace2005_to_unified(args.ACE_DUMP, "train.json")
    convert_ace2005_to_unified(args.ACE_DUMP, "valid.json")
    convert_ace2005_to_unified(args.ACE_DUMP, "test.json")

