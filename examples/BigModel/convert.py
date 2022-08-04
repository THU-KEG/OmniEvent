from collections import OrderedDict
import torch
import os 
from pathlib import Path 
from tqdm import tqdm
from model_center.model import T5
import bmtrain as bmt
import torch 

def check_nan(state_dict):
    for name, param in state_dict.items():
        if torch.isnan(param).sum() != 0:
            print(name)

def main():
    ver_layernum = [
        ("small",8),
        ("base", 12),
        ("large", 24),
        ("xl", 24),
        ("xxl", 24),
    ]

    for n in [2]:
        ver, layernum = ver_layernum[n]

        # out = torch.load(f"/ldata/ph/ModelCenter-Model/mt5-{ver}/pytorch_model.pt")
        out = torch.load(f"results/finetune-mt5-{ver}-ckpt-9.pt")
        check_nan(out)

        outdir = Path(f"output/mt5-{ver}/")
        outdir.mkdir(exist_ok=True, parents=True)
        outpath = os.path.join(outdir, "pytorch_model.bin")
        inp = OrderedDict()

        inp["lm_head.weight"] = out["output_projection.weight"].contiguous()
        inp["encoder.embed_tokens.weight"] = out["input_embedding.weight"].contiguous()
        inp["decoder.embed_tokens.weight"] = out["input_embedding.weight"].contiguous()
        inp["shared.weight"] = out["input_embedding.weight"].contiguous()

        inp["encoder.final_layer_norm.weight"] = out["encoder.output_layernorm.weight"].contiguous()
        inp["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = out["position_bias_enc.relative_attention_bias"].contiguous()
        inp["decoder.final_layer_norm.weight"] = out["decoder.output_layernorm.weight"].contiguous()
        inp["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = out["position_bias_dec.relative_attention_bias"].contiguous()

        for i in range(layernum):
            prefix = f"encoder.layers.{i}"
            old_prefix = f"encoder.block.{i}"

            inp[f"{old_prefix}.layer.0.layer_norm.weight"] = out[f"{prefix}.self_att.layernorm_before_attention.weight"].contiguous()
            inp[f"{old_prefix}.layer.0.SelfAttention.q.weight"] = out[f"{prefix}.self_att.self_attention.project_q.weight"].contiguous() #[:attn_project_size]
            inp[f"{old_prefix}.layer.0.SelfAttention.k.weight"] = out[f"{prefix}.self_att.self_attention.project_k.weight"].contiguous() #[attn_project_size:2*attn_project_size]
            inp[f"{old_prefix}.layer.0.SelfAttention.v.weight"] = out[f"{prefix}.self_att.self_attention.project_v.weight"].contiguous() #[2*attn_project_size:]
            inp[f"{old_prefix}.layer.0.SelfAttention.o.weight"] = out[f"{prefix}.self_att.self_attention.attention_out.weight"].contiguous()

            inp[f"{old_prefix}.layer.1.layer_norm.weight"] = out[f"{prefix}.ffn.layernorm_before_ffn.weight"].contiguous()
            inp[f"{old_prefix}.layer.1.DenseReluDense.wi_0.weight"] = out[f"{prefix}.ffn.ffn.w_in.w_0.weight"].contiguous()
            inp[f"{old_prefix}.layer.1.DenseReluDense.wi_1.weight"] = out[f"{prefix}.ffn.ffn.w_in.w_1.weight"].contiguous()
            inp[f"{old_prefix}.layer.1.DenseReluDense.wo.weight"] = out[f"{prefix}.ffn.ffn.w_out.weight"].contiguous()

        for i in range(layernum):
            prefix = f"decoder.layers.{i}"
            old_prefix = f"decoder.block.{i}"
            inp[f"{old_prefix}.layer.0.layer_norm.weight"] = out[f"{prefix}.self_att.layernorm_before_attention.weight"].contiguous()
            inp[f"{old_prefix}.layer.0.SelfAttention.q.weight"] = out[f"{prefix}.self_att.self_attention.project_q.weight"].contiguous() #[:attn_project_size]
            inp[f"{old_prefix}.layer.0.SelfAttention.k.weight"] = out[f"{prefix}.self_att.self_attention.project_k.weight"].contiguous() #[attn_project_size:2*attn_project_size]
            inp[f"{old_prefix}.layer.0.SelfAttention.v.weight"] = out[f"{prefix}.self_att.self_attention.project_v.weight"].contiguous() #[2*attn_project_size:]
            inp[f"{old_prefix}.layer.0.SelfAttention.o.weight"] = out[f"{prefix}.self_att.self_attention.attention_out.weight"].contiguous()

            inp[f"{old_prefix}.layer.1.layer_norm.weight"] = out[f"{prefix}.cross_att.layernorm_before_attention.weight"].contiguous()
            inp[f"{old_prefix}.layer.1.EncDecAttention.q.weight"] = out[f"{prefix}.cross_att.self_attention.project_q.weight"].contiguous()
            inp[f"{old_prefix}.layer.1.EncDecAttention.k.weight"] = out[f"{prefix}.cross_att.self_attention.project_k.weight"].contiguous() #[:attn_project_size]
            inp[f"{old_prefix}.layer.1.EncDecAttention.v.weight"] = out[f"{prefix}.cross_att.self_attention.project_v.weight"].contiguous() #[attn_project_size:]
            inp[f"{old_prefix}.layer.1.EncDecAttention.o.weight"] = out[f"{prefix}.cross_att.self_attention.attention_out.weight"].contiguous()

            inp[f"{old_prefix}.layer.2.layer_norm.weight"] = out[f"{prefix}.ffn.layernorm_before_ffn.weight"].contiguous()
            inp[f"{old_prefix}.layer.2.DenseReluDense.wi_0.weight"] = out[f"{prefix}.ffn.ffn.w_in.w_0.weight"].contiguous()
            inp[f"{old_prefix}.layer.2.DenseReluDense.wi_1.weight"] = out[f"{prefix}.ffn.ffn.w_in.w_1.weight"].contiguous()
            inp[f"{old_prefix}.layer.2.DenseReluDense.wo.weight"] = out[f"{prefix}.ffn.ffn.w_out.weight"].contiguous()

        # for k, v in out.items():
            # out[k] = out[k].half()

        torch.save(inp, outpath)
        indir = f"/ldata/ph/OpenEE/examples/EAE/output/ALL-EAE/EAE/seq2seq/mt5-{ver}/forbm"
        os.system(f"cp {indir}/added_tokens.json {indir}/tokenizer.json {indir}/config.json {indir}/special_tokens_map.json {indir}/tokenizer_config.json {outdir}")

if __name__ == "__main__":
    main()