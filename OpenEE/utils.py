import os 
import json 
import requests
from tqdm import tqdm 


MODEL_NAMES = {
    "tc-bert-marker-linear": "",
    "tc-bert-cls-linear": "",
    "tc-roberta-marker-linear": "",
    "tc-roberta-cls-linear": "",
    "sl-bert-none-none": "",
    "sl-bert-none-crf": "",
    "s2s-t5": "",
    "s2s-mt5": "",
    "s2s-bart": ""
}

FILE_NAMES = {
    'config': ['config.yaml'],
    'model': ['pytorch_model.bin'],
    'tokenizer': ['vocab.json', 'vocab.txt', 'merges.txt', 'tokenizer.json', 'added_tokens.json', 'special_tokens_map.json', 'tokenizer_config.json', 'spiece.model', 'vocab.model'],
}


def download(path, url):
    req = requests.get(url, stream=True)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file = open(path, "wb")
        req.raise_for_status()
        print(f"download from web, cache will be save to: {path}")
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm.tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=total,
            desc="Downloading",
        )
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                file.write(chunk)
        progress.close()
        file.close()
    except:
        file.close()
        os.remove(path)


def check_web_and_convert_path(path, load_type, base_path="OpenEE"): # TODO add hash
    if os.path.isdir(path):
        print(f"load from local file: {path}")
        return path
    else:
        if path not in MODEL_NAMES:
            raise ValueError(f"'{path}' is not a valid model identifier")
        url = MODEL_NAMES[path]
        try:
            requests.get(f'{url}/config.json', stream=True).raise_for_status() # use config.json to check if identifier is valid
        except:
            raise ValueError(f"'{path}' is not a valid model identifier")
        cache_path = os.path.expanduser(f"{base_path}/{path}")
        for name in FILE_NAMES[load_type]:
            p = os.path.join(cache_path, name)
            if os.path.exists(p):
                print(f"load from cache: {p}")
            else:
                download(p, url)
        else:
            cache_path = os.path.expanduser(f"{base_path}/{path}")
        return cache_path


