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
        ("small",6),
        ("base", 12),
        ("large", 24),
        ("3b", 24),
        ("11b", 24),
    ]
    ver, layernum = ver_layernum[4]
    inpath = f"../results/t5-{ver}-pytorch_model.bin"
    outpath = f"../results/T5-{ver}.pt"
    scale = 100
    inp = torch.load(inpath)
    out = OrderedDict()
    out["input_embedding.weight"] = inp["shared.weight"].contiguous()
    with torch.no_grad(): out["input_embedding.weight"] /= scale
    out["encoder.output_layernorm.weight"] = inp["encoder.final_layer_norm.weight"].contiguous()
    out["position_bias_enc.relative_attention_bias"] = inp["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].contiguous()
    out["decoder.output_layernorm.weight"] = inp["decoder.final_layer_norm.weight"].contiguous()
    out["position_bias_dec.relative_attention_bias"] = inp["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].contiguous()
    for i in range(layernum):
        prefix = f"encoder.layers.{i}"
        old_prefix = f"encoder.block.{i}"
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layer.0.layer_norm.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.q.weight"].contiguous() #[:attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.k.weight"].contiguous() #[attn_project_size:2*attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.v.weight"].contiguous() #[2*attn_project_size:]
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.o.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.self_att.self_attention.attention_out.weight"] /= scale

        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.layer.1.layer_norm.weight"].contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.layer.1.DenseReluDense.wi.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_in.w.weight"] /= scale**0.5
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.layer.1.DenseReluDense.wo.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_out.weight"] /= scale**0.5

    for i in range(layernum):
        prefix = f"decoder.layers.{i}"
        old_prefix = f"decoder.block.{i}"
        out[f"{prefix}.self_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layer.0.layer_norm.weight"].contiguous()
        out[f"{prefix}.self_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.q.weight"].contiguous() #[:attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.k.weight"].contiguous() #[attn_project_size:2*attn_project_size]
        out[f"{prefix}.self_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.v.weight"].contiguous() #[2*attn_project_size:]
        out[f"{prefix}.self_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.layer.0.SelfAttention.o.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.self_att.self_attention.attention_out.weight"] /= scale

        out[f"{prefix}.cross_att.layernorm_before_attention.weight"] = inp[f"{old_prefix}.layer.1.layer_norm.weight"].contiguous()
        out[f"{prefix}.cross_att.self_attention.project_q.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.q.weight"].contiguous()
        out[f"{prefix}.cross_att.self_attention.project_k.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.k.weight"].contiguous() #[:attn_project_size]
        out[f"{prefix}.cross_att.self_attention.project_v.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.v.weight"].contiguous() #[attn_project_size:]
        out[f"{prefix}.cross_att.self_attention.attention_out.weight"] = inp[f"{old_prefix}.layer.1.EncDecAttention.o.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.cross_att.self_attention.attention_out.weight"] /= scale

        out[f"{prefix}.ffn.layernorm_before_ffn.weight"] = inp[f"{old_prefix}.layer.2.layer_norm.weight"].contiguous()
        out[f"{prefix}.ffn.ffn.w_in.w.weight"] = inp[f"{old_prefix}.layer.2.DenseReluDense.wi.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_in.w.weight"] /= scale**0.5
        out[f"{prefix}.ffn.ffn.w_out.weight"] = inp[f"{old_prefix}.layer.2.DenseReluDense.wo.weight"].contiguous()
        with torch.no_grad(): out[f"{prefix}.ffn.ffn.w_out.weight"] /= scale**0.5

    for k, v in out.items():
        out[k] = out[k].half()

    torch.save(out, outpath)

if __name__ == "__main__":
    main()
