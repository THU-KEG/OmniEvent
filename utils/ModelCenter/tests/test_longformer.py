#coding:utf-8

from cgitb import lookup
import torch
import bmtrain as bmt
from model_center.tokenizer import BertTokenizer
from model_center.model import Longformer
from transformers import LongformerForMaskedLM
from transformers import BertForMaskedLM as hugBert

import sys
def main():
    bmt.init_distributed()

    bmt_bert = Longformer.from_pretrained("lawformer")
    hug_bert = LongformerForMaskedLM.from_pretrained("thunlp/Lawformer").cuda()
    bmt_bert.eval()
    hug_bert.eval()
    b_emb=bmt_bert._modules['input_embedding']
    h_emb=hug_bert._modules['longformer']._modules['embeddings']._modules['word_embeddings']
    for i in range(1): 
        batch = 1
        max_encoder_length = 2048
        input_ids = torch.randint(21128, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]
        global_attn = torch.zeros(input_ids.shape[1],device=input_ids.device).repeat(input_ids.shape[0], 1)
        global_attn[:,:100] = 1
        bmt_logits = bmt_bert(input_ids = input_ids, return_logits=True,attention_mask=attention_mask,global_attention_mask=global_attn)
        hug_logits = hug_bert(input_ids = input_ids,attention_mask=attention_mask,global_attention_mask=global_attn).logits
        b = bmt_logits*attention_mask[:,:,None]
        h = hug_logits*attention_mask[:,:,None]
        d = (h - b).abs()
        emb_grad={}
        print(d.max())
        def hook(name):
            def backward_hook(module, grad_input, grad_output):
                emb_grad[name]=grad_output[0]
            return backward_hook
        h_emb.register_full_backward_hook(hook("h"))
        b_emb.register_full_backward_hook(hook("b"))
        loss_func = torch.nn.CrossEntropyLoss()
        labels=torch.randint(21128, (batch, max_encoder_length,), dtype=torch.long).cuda()
        loss1 = loss_func(b.view(-1,b.shape[-1]), labels.view(-1))
        loss2 = loss_func(h.view(-1,h.shape[-1]), labels.view(-1))
        loss1.backward()
        loss2.backward()
        if i>0:
            d_grad=(emb_grad["h"]-emb_grad["b"]).abs()
            print(d_grad.max())
if __name__ == "__main__":
    main()
