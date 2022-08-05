import os 
import json 
from .arguments import (
    ArgumentParser, 
    ModelArguments,
    DataArguments,
    TrainingArguments
)
from .model.model import get_model_cls
from .input_engineering.whitespace_tokenizer import WhitespaceTokenizer
from .utils import check_web_and_convert_path
from transformers import (
    BertTokenizerFast,
    RobertaTokenizerFast,
    T5TokenizerFast,
    MT5TokenizerFast,
    BartTokenizerFast
)

TOKENIZER_NAME_TO_CLS = {
    "BertTokenizer": BertTokenizerFast,
    "RobertaTokenizer": RobertaTokenizerFast,
    "T5Tokenizer": T5TokenizerFast,
    "MT5Tokenizer": MT5TokenizerFast,
    "BartTokenizer": BartTokenizerFast
}


def get_tokenizer(tokenizer_name_or_path):
    path = check_web_and_convert_path(tokenizer_name_or_path, "tokenizer")
    tokenizer_config = json.load(os.path.join(path, "tokenizer_config.json"))
    tokenizer_cls = TOKENIZER_NAME_TO_CLS[tokenizer_config["tokenizer_class"]]
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_name_or_path)
    return tokenizer_name_or_path
    

def get_pretrained(model_name_or_path):
    # config 
    parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.from_pretrained(model_name_or_path)
    # model
    model = get_model_cls(model_args).from_pretrained(model_name_or_path)
    # tokenizer 
    tokenizer = get_tokenizer(model_name_or_path)

    return model, tokenizer, (model_args, data_args, training_args)


def infer(model, tokenizer, task="ED"):
    pass 
