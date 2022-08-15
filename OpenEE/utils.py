import os 
import json 
import requests
import tqdm 


MODEL_NAMES = {
    "s2s-mt5-ed": "https://cloud.tsinghua.edu.cn/f/cdc4b333aff143ff870e/?dl=1",
    "s2s-mt5-eae": ""
}

FILE_NAMES = {
    'config': ['config.json'],
    'model': ['pytorch_model.bin'],
    'tokenizer': ['vocab.json', 'vocab.txt', 'merges.txt', 'tokenizer.json', 'added_tokens.json', 'special_tokens_map.json', 'tokenizer_config.json', 'spiece.model', 'vocab.model'],
    'args': ['args.yaml']
}


def download(path, url):
    req = requests.get(url, stream=True)
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
    print(total)
    print("Downloading")
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            file.write(chunk)
    progress.close()
    file.close()
    os.system(f"unzip {path}")
    os.system(f"rm {path}")


def check_web_and_convert_path(path, load_type, base_path="OpenEE-Model"): # TODO add hash
    if os.path.isdir(path):
        print(f"load from local file: {path}")
        return path
    if os.path.isdir(os.path.join(base_path, path)):
        print(f"load from local file: {os.path.join(base_path, path)}")
        return os.path.join(base_path, path)
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    else:
        if path not in MODEL_NAMES:
            raise ValueError(f"'{path}' is not a valid model identifier")
        url = MODEL_NAMES[path]
        try:
            requests.get(url, stream=True).raise_for_status() # use config.json to check if identifier is valid
        except:
            raise ValueError(f"'{path}' is not a valid model identifier")
        cache_path = f"{base_path}/{path}"
        download(cache_path+".zip", url)
        return cache_path


