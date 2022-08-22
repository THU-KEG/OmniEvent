#coding:utf-8

import torch
import bmtrain as bmt
from model_center.model.config import RobertaConfig
from model_center.model import Roberta
from model_center.tokenizer import RobertaTokenizer

from transformers import BertTokenizer, RobertaForMaskedLM as hugRoberta

def main():
    bmt.init_distributed()

    # path = "roberta-base"
    path = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(path)
    config = RobertaConfig.from_pretrained(path)
    config.dropout_p = 0
    bmt_roberta = Roberta.from_pretrained(path, config=config)

    hug_roberta = hugRoberta.from_pretrained(path).cuda().eval().half()

    for _ in range(10):
        batch = 1
        max_encoder_length = 512
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]

        bmt_logits = bmt_roberta(input_ids = input_ids, attention_mask = attention_mask, return_logits=True)
        hug_logits = hug_roberta(input_ids = input_ids, attention_mask = attention_mask).logits
        b = bmt_logits*attention_mask[:,:,None]
        h = hug_logits*attention_mask[:,:,None]
        d = (h - b).abs()
        print(d.max())

if __name__ == "__main__":
    main()
