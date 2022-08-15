#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import BertTokenizer
from model_center.model import BertConfig, Bert
from transformers import BertModel as hugBert
from model_center.model import GPT2Config, GPT2
from transformers import GPT2Model as hugGPT2


def main():
    bmt.init_distributed()

    # path = "bert-base-uncased"
    # config = BertConfig.from_pretrained(path)
    # config.dropout_p = 0
    # bmt_bert = Bert.from_pretrained(path, config=config)
    # hug_bert = hugBert.from_pretrained(path).cuda().eval().half()

    path = "gpt2-base"
    config = GPT2Config.from_pretrained(path, use_cache = True)
    config.dropout_p = 0
    bmt_bert = GPT2.from_pretrained(path, config=config)
    hug_bert = hugGPT2.from_pretrained('gpt2').cuda().eval().half()

    cur_len = 0
    add_len = 1
    bmt_pkv = None
    hug_pkv = None

    input_ids_list = []
    bmt_logits_list = []
    hug_logits_list = []

    for _ in range(100):

        batch = 2
        input_ids = torch.randint(config.vocab_size, (batch, add_len,), dtype=torch.int32).cuda()
        attention_mask = torch.ones((batch, add_len + cur_len, add_len), dtype=torch.int32).cuda()
        attention_mask_1 = torch.ones(((batch, add_len + cur_len)), dtype=torch.int32).cuda()

        bmt_res = bmt_bert(input_ids = input_ids, attention_mask = attention_mask, use_cache = True, past_key_values = bmt_pkv)
        bmt_pkv = bmt_res.past_key_values
        bmt_logits = bmt_res.last_hidden_state
        bmt_logits_list.append(bmt_logits)

        input_ids_list.append(input_ids)
        hug_res = hug_bert(input_ids = input_ids, attention_mask = attention_mask_1, use_cache = True, past_key_values = hug_pkv)
        hug_pkv = hug_res.past_key_values        
        hug_logits = hug_res.last_hidden_state
        hug_logits_list.append(hug_logits)

        cur_len += add_len

    bmt_logits_pkv = torch.cat(bmt_logits_list, dim=1)
    hug_logits_pkv = torch.cat(hug_logits_list, dim=1)
    print((bmt_logits_pkv - hug_logits_pkv).abs().mean())

    input_ids = torch.cat(input_ids_list, dim=1)
    logits = bmt_bert(input_ids = input_ids, attention_mask = torch.ones((2, cur_len), dtype=torch.int32).cuda()).last_hidden_state
    print((logits - bmt_logits_pkv).abs().mean())

if __name__ == "__main__":
    main()
