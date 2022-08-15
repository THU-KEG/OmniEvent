#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import BertTokenizer
from model_center.model import BertConfig, Bert

from transformers import BertModel as hugBert

def main():
    bmt.init_distributed()

    path = "bert-base-uncased"
    config = BertConfig.from_pretrained(path)
    config.dropout_p = 0
    bmt_bert = Bert.from_pretrained(path, config=config)

    cur_len = 0
    add_len = 8
    bmt_pkv = None
    hug_pkv = None

    input_ids_list = []
    logits_list = []
    attention_mask_all = None

    for _ in range(40):
        batch = 2
        input_ids = torch.randint(config.vocab_size, (batch, add_len,), dtype=torch.int32).cuda()
        attention_mask = torch.randint(2,(batch, add_len, add_len + cur_len), dtype=torch.int32).cuda()

        bmt_res = bmt_bert(input_ids = input_ids, attention_mask = attention_mask, use_cache = True, past_key_values = bmt_pkv)
        bmt_pkv = bmt_res.past_key_values
        bmt_logits = bmt_res.last_hidden_state

        input_ids_list.append(input_ids)
        logits_list.append(bmt_logits)
        if attention_mask_all is None:
            attention_mask_all = attention_mask
        else:
            attention_mask_all = torch.cat([attention_mask_all, torch.zeros(batch, cur_len, add_len).cuda()], dim=2)
            attention_mask_all = torch.cat([attention_mask_all, attention_mask], dim=1)

        cur_len += add_len

    input_ids = torch.cat(input_ids_list, dim=1)
    logits_pkv = torch.cat(logits_list, dim=1)
    logits = bmt_bert(input_ids = input_ids, attention_mask = attention_mask_all).last_hidden_state
    print((logits - logits_pkv).abs().max())

if __name__ == "__main__":
    main()
