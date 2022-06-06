import os 
import re 
import pdb 

import torch 

from collections import defaultdict
from.input_utils import load_ontology



class Seq2SeqBase:
    def __init__(self) -> None:
        pass

    def serialize(self, arguments, ontology):
        raise NotImplementedError
    
    def deserialize(self, prediction_text, template, ontology):
        raise NotImplementedError

    def extract_from_text(self, decoded_texts, event_types):
        raise NotImplementedError
    
    def convert_to_final_list(self, labels, preds, true_types, pred_types):
        raise NotImplementedError



class Seq2SeqTemplate(Seq2SeqBase):
    def __init__(self, ontology_file) -> None:
        self.ontology_dict = load_ontology(ontology_file)
    
    def serialize(self, arguments, ontology):
        template = ontology["template"]
        # input template 
        input_template = re.sub(r"-arg\d-", "-arg-", template)

        # output tempalte 
        role2arg = defaultdict(list)
        for argument in arguments:
            for mention in argument["mentions"]:
                role2arg[argument["role"]].append(mention)
        
        arg_idx2text = defaultdict(list)
        for role in role2arg.keys():
            if role not in ontology:
                # annotation error 
                continue
            for i, mention in enumerate(role2arg[role]):
                arg_text = mention["mention"]
                if i < len(ontology[role]):
                    # enough slots to fill in 
                    arg_idx = ontology[role][i]
                else:
                    # multiple participants for the same role 
                    arg_idx = ontology[role][-1]
                arg_idx2text[arg_idx].append(arg_text)
        # serialize 
        for arg_idx, text_list in arg_idx2text.items():
            text = ' AND '.join(text_list)
            template = re.sub('-{}-'.format(arg_idx), text, template)
        output_template = re.sub(r'-arg\d-','-arg-', template) 

        return input_template, output_template

    def deserialize(self, prediction_text, template, ontology):
        # extract argument text 
        template_words = template.strip().split()
        predicted_words = prediction_text.strip().split()    
        predicted_args = defaultdict(list)
        t_ptr= 0
        p_ptr= 0 

        while t_ptr < len(template_words) and p_ptr < len(predicted_words):
            if re.match(r'-(arg\d+)-', template_words[t_ptr]):
                m = re.match(r'-(arg\d+)-', template_words[t_ptr])
                arg_num = m.group(1)
                arg_name = ontology[arg_num]

                if predicted_words[p_ptr] == '-arg-':
                    # missing argument
                    p_ptr +=1 
                    t_ptr +=1  
                else:
                    arg_start = p_ptr 
                    while (p_ptr < len(predicted_words)) and (t_ptr==len(template_words)-1 or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                        p_ptr+=1 
                    arg_text = predicted_words[arg_start:p_ptr]
                    predicted_args[arg_name].extend(self.extract_multiple_mention(arg_text))
                    t_ptr+=1 
                    # aligned 
            else:
                t_ptr+=1 
                p_ptr+=1 

        return dict(predicted_args)
    

    def extract_multiple_mention(self, tokens):
        mentions = []
        mention = ""
        for token in tokens:
            if token == "AND":
                mentions.append(mention.strip())
                mention = ""
                continue
            mention += token + " "
        if mention != "":
            mentions.append(mention.strip())
        return mentions 


    def extract_from_text(self, decoded_texts, event_types):
        assert len(decoded_texts) == len(event_types)
        all_args = []
        for text, event_type in zip(decoded_texts, event_types):
            ontology = self.ontology_dict[event_type]
            template = ontology["template"]
            args = self.deserialize(text, template, ontology)
            all_args.append(args)
        return all_args

    
    def convert_to_final_list(self, labels, preds, true_types, pred_types):
        final_labels = []
        final_preds = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            if true_types[i] != pred_types[i]:
                for role, mentions in label.items():
                    for mention in mentions:
                        final_labels.append(role)
                        final_preds.append("NA")
            else:
                for role in label.keys():
                    golden_mentions = label[role]
                    pred_mentions = pred.get(role, [])
                    for mention in golden_mentions:
                        final_labels.append(role)
                        if mention in pred_mentions:
                            final_preds.append(role)
                        else:
                            final_preds.append("NA")
                    for mention in pred_mentions:
                        if mention not in golden_mentions:
                            final_preds.append(role)
                            final_labels.append("NA")
        assert len(final_labels) == len(final_preds)
        return final_labels, final_preds
    


