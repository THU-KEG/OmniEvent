# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
import torch
from tqdm import tqdm

def main():
    ver_layernum = [
        ("base", 12),
        ("medium", 24),
        ("large", 36),
        ("xl", 48),
    ]
    ver, layernum = ver_layernum[0]
    inpath = f"../results/gpt2-{ver}-pytorch_model.bin"
    outpath = f"../results/GPT2-{ver}.pt"
    inp = torch.load(inpath)
    out = OrderedDict()
    out["input_embedding.weight"] = torch.cat([inp["wte.weight"], torch.zeros((1,inp["wte.weight"].shape[1]))], dim=0).contiguous() # original vocab size is an odd number
    out["position_embedding.weight"] = inp["wpe.weight"].contiguous()
    out["encoder.output_layernorm.weight"] = inp["ln_f.weight"].contiguous()
    out["encoder.output_layernorm.bias"] = inp["ln_f.bias"].contiguous()
    for i in range(layernum):
        prefix = f"encoder.layers.{i}"
        old_prefix = f"h.{i}"
        attn_size = inp[f"{old_prefix}.attn.c_attn.weight"].shape[0]
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.ln_1.weight"].contiguous()
        out[f"{prefix}.self_att.layernorm_before_attention.bias"] = inp[f"{old_prefix}.ln_1.bias"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.attn.c_attn.weight"][:, :attn_size].transpose(0,1).contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.bias"] = inp[f"{old_prefix}.attn.c_attn.bias"][:attn_size].contiguous()
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.attn.c_attn.weight"][:, attn_size:2*attn_size].transpose(0,1).contiguous()
        out[f"{prefix}.self_att.self_attention.project_k.bias"] = inp[f"{old_prefix}.attn.c_attn.bias"][attn_size:2*attn_size].contiguous()
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.attn.c_attn.weight"][:, 2*attn_size:].transpose(0,1).contiguous()
        out[f"{prefix}.self_att.self_attention.project_v.bias"] = inp[f"{old_prefix}.attn.c_attn.bias"][2*attn_size:].contiguous()
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.attn.c_proj.weight"].transpose(0,1).contiguous()
        out[f"{prefix}.self_att.self_attention.attention_out.bias"] = inp[f"{old_prefix}.attn.c_proj.bias"].contiguous()

        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.ln_2.weight"].contiguous()
        out[f"{prefix}.ffn.layernorm_before_ffn.bias"] = inp[f"{old_prefix}.ln_2.bias"].contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.mlp.c_fc.weight"].transpose(0,1).contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.bias"] = inp[f"{old_prefix}.mlp.c_fc.bias"].contiguous()
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.mlp.c_proj.weight"].transpose(0,1).contiguous()
        out[f"{prefix}.ffn.ffn.w_out.bias"] = inp[f"{old_prefix}.mlp.c_proj.bias"].contiguous()

    for k, v in out.items():
        out[k] = out[k].half()

    torch.save(out, outpath)

if __name__ == "__main__":
    main()
