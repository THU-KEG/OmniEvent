import torch 
import transformers 

from transformers import BertModel, BertTokenizer, BertTokenizerFast 
from transformers import RobertaModel, RobertaTokenizer, RobertaTokenizerFast


def get_backbone(model_type, model_name_or_path, tokenizer_name, markers, new_tokens:list = []):
    if model_type == "bert":
        model = BertModel.from_pretrained(model_name_or_path)
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    elif model_type == "roberta":
        model = RobertaModel.from_pretrained(model_name_or_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, never_split=markers)
    else:
        raise ValueError("No such model. %s" % model_type)
    
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    config = model.config
    return model, tokenizer, config

