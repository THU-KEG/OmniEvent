#coding:utf-8

import torch
import bmtrain as bmt

from model_center.model import ViT,VitConfig
from transformers import  ViTForImageClassification



def main():
    bmt.init_distributed()

    path = "vit-base_patch16_224"
    config = VitConfig.from_pretrained(path)
    config.dropout_p = 0
    bmt_vit = ViT.from_pretrained(path, config=config)
    hug_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').cuda().half()
    def hook(name):
        def backward_hook(module, grad_input, grad_output):
            emb_grad[name]=grad_output[0]
        return backward_hook
    for i in range(10):
        with torch.autograd.set_detect_anomaly(True): 
            batch = 12
            # max_encoder_length = 512
            patch_size=224
            channel_size=3
            # inputs  =  torch.randn((1,3,224,224),dtype = torch.half).cuda().to(memory_format=torch.channels_last)
            inputs  =  torch.randn((batch,channel_size,patch_size,patch_size),dtype = torch.half).cuda().to(memory_format=torch.channels_last)
            
            bmt_logits = bmt_vit(inputs)
            hug_logits = hug_vit(inputs).logits
            b = bmt_logits
            h = hug_logits
            d = (h - b).abs()
            print(d.max())
            b_emb=bmt_vit.patch_embed.proj
            h_emb=hug_vit.vit.embeddings.patch_embeddings.projection
            emb_grad={}
            h_emb.register_full_backward_hook(hook("h"))
            b_emb.register_full_backward_hook(hook("b"))
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
            labels=torch.randint(1000, (batch,), dtype=torch.long).cuda()
            loss1 = loss_func(b, labels)
            loss2 = loss_func(h, labels)
            loss1.backward()
            loss2.backward()
            if i>0:
                d_grad=(emb_grad["h"]-emb_grad["b"]).abs()
                print(d_grad.max())
if __name__ == "__main__":
    main()
