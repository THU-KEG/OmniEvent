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
from .infer_module.seq2seq import (
    do_event_detection, 
    do_event_argument_extraction,
    get_ed_result,
    get_eae_result,
    prepare_for_eae_from_input,
    prepare_for_eae_from_pred
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
    tokenizer_config = json.load(open(os.path.join(path, "tokenizer_config.json")))
    tokenizer_cls = TOKENIZER_NAME_TO_CLS[tokenizer_config["tokenizer_class"]]
    tokenizer = tokenizer_cls.from_pretrained(path)
    return tokenizer


def get_model(model_args, model_name_or_path):
    path = check_web_and_convert_path(model_name_or_path, "model")
    model = get_model_cls(model_args).from_pretrained(path)
    return model
    

def get_pretrained(model_name_or_path):
    # config 
    parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.from_pretrained(model_name_or_path)
    # model
    model = get_model(model_args, model_name_or_path)
    # tokenizer 
    tokenizer = get_tokenizer(model_name_or_path)

    return model, tokenizer, (model_args, data_args, training_args)


def infer(text, triggers=None, task="ED"):
    if task == "ED":
        ed_model, ed_tokenizer, _ = get_pretrained("s2s-mt5-ed")
        ed_model.cuda()
        events = do_event_detection(ed_model, ed_tokenizer, [text])
        results = get_ed_result([text], events)
    elif task == "EAE":
        eae_model, eae_tokenizer, _ = get_pretrained("s2s-mt5-eae")
        instances = prepare_for_eae_from_input([text], triggers)
        arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances)
        results = get_eae_result(instances, arguments)
    elif task == "EE":
        ed_model, ed_tokenizer, _ = get_pretrained("s2s-mt5-ed")
        eae_model, eae_tokenizer, _ = get_pretrained("s2s-mt5-eae")
        events = do_event_detection(ed_model, ed_tokenizer, [text])
        instances = prepare_for_eae_from_pred([text], events)
        if len(instances[0]["triggers"]) == 0:
            results = [{
                "text": instances[0]["text"],
                "events": []
            }]
            return results
        arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances)
        results = get_eae_result(instances, arguments)
    print(results)
    return results
