#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import GPTjTokenizer
from model_center.model import GPTjConfig, GPTj

from transformers import GPTJForCausalLM as hugGPTj

def main():
    bmt.init_distributed()

    tokenizer = GPTjTokenizer.from_pretrained("gptj-6b")
    config = GPTjConfig.from_pretrained("gptj-6b")
    config.dropout_p = 0
    bmt_gptj = GPTj.from_pretrained("gptj-6b")

    hug_gptj = hugGPTj.from_pretrained("EleutherAI/gpt-j-6B").cuda().eval().half()
    
    for _ in range(10):
        batch = 1
        max_encoder_length = 512
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]

        bmt_logits = bmt_gptj(input_ids = input_ids, attention_mask = attention_mask, output_logits=True).logits
        hug_logits = hug_gptj(input_ids = input_ids, attention_mask = attention_mask).logits
        b = (bmt_logits*attention_mask[:,:,None])
        h = hug_logits*attention_mask[:,:,None]
        d = (h - b).abs()
        print(d.max())

if __name__ == "__main__":
    main()
