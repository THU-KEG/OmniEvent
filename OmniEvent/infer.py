import os
import json

import torch.cuda

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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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


def get_pretrained(model_name_or_path, device):
    # model
    model_args = AttrDict({
        "paradigm": "seq2seq",
        "model_type": "mt5"
    })
    model = get_model(model_args, model_name_or_path)
    model = model.to(device)
    # tokenizer 
    tokenizer = get_tokenizer(model_name_or_path)

    return model, tokenizer


def infer(text, model=None, tokenizer=None, triggers=None, schema="ace", task="ED", device='auto'):
    """Infer method.

    Args:
        text (`str`): Input plain text.
        triggers (`List[List]`, *optional*): List of triggers in the text. Only useful for EAE. Examples: [(moving, 2, 8), ...]
        schema (`str`): Schema used for ED and EAE. Selected in ['ace', 'kbp', 'ere', 'maven', 'leven', 'duee', 'fewfc']
        task (`str`): Task type. Selected in ['ED', 'EAE', 'EE']
    
    Returns:
        results (`List`): Predicted results. The format is 
        [
            {
                'text': `text`, 
                'events': [
                    {
                        'type': `type`,
                        'trigger': `trigger word`,
                        'offset': [`char start`, `char end`],
                        'arguments': [ // for EAE and EE
                            {
                                'mention': `argument mention`,
                                'offset': [`char start`, `char end`],
                                'role': `argument role`
                            }
                        ]
                    }
                ]
            } 
        ]
    """
    assert schema in ['ace', 'kbp', 'ere', 'maven', 'leven', 'duee', 'fewfc']
    assert task in ['ED', 'EAE', 'EE']
    schema = f"<{schema}>"
    # get device.
    if device == 'auto':
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = 'cuda'
    else:
        device = torch.device(device)

    if task == "ED":
        if model is None or tokenizer is None:
            ed_model, ed_tokenizer = get_pretrained("s2s-mt5-ed", device)
        else:
            ed_model, ed_tokenizer = model, tokenizer
        events = do_event_detection(ed_model, ed_tokenizer, [text], [schema], device)
        results = get_ed_result([text], events)
    elif task == "EAE":
        if model is None or tokenizer is None:
            eae_model, eae_tokenizer = get_pretrained("s2s-mt5-eae", device)
        else:
            eae_model, eae_tokenizer = model, tokenizer
        instances = prepare_for_eae_from_input([text], [triggers], [schema])
        arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances, device)
        results = get_eae_result(instances, arguments)
    elif task == "EE":
        if model is None or tokenizer is None:
            ed_model, ed_tokenizer = get_pretrained("s2s-mt5-ed", device)
            eae_model, eae_tokenizer = get_pretrained("s2s-mt5-eae", device)
        else:
            ed_model, ed_tokenizer = model[0], tokenizer[0]
            eae_model, eae_tokenizer = model[1], tokenizer[1]
        events = do_event_detection(ed_model, ed_tokenizer, [text], [schema], device)
        instances = prepare_for_eae_from_pred([text], events, [schema])
        if len(instances[0]["triggers"]) == 0:
            results = [{
                "text": instances[0]["text"],
                "events": []
            }]
            return results
        arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances, device)
        results = get_eae_result(instances, arguments)
    print(results)
    return results
