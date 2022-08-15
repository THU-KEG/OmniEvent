#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import T5Tokenizer
from model_center.model import T5Config, T5

from transformers import T5ForConditionalGeneration as hugT5

def main():
    bmt.init_distributed()

    path = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(path)
    config = T5Config.from_pretrained(path)
    config.scale = True
    bmt_t5 = T5.from_pretrained(path, config=config)

    hug_t5 = hugT5.from_pretrained(path).cuda()
    
    for _ in range(10):
        batch = 1
        max_encoder_length = 512
        max_decoder_length = 512
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        decoder_input_ids = torch.randint(config.vocab_size, (batch, max_decoder_length,), dtype=torch.int32).cuda()
        decoder_length = torch.randint(max_decoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]
        decoder_attention_mask = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)[None, :].repeat(decoder_input_ids.shape[0], 1) < decoder_length[:, None]

        bmt_logits = bmt_t5(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, output_logits=True).logits
        hug_logits = hug_t5(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask).logits
        b = bmt_logits*decoder_attention_mask[:,:,None]
        h = hug_logits*decoder_attention_mask[:,:,None]
        d = (h - b).abs()
        print(d.max())
        print(h / b)

if __name__ == "__main__":
    main()
