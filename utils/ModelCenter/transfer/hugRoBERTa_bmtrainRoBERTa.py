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

import os
import torch
import json

from collections import OrderedDict
from transformers import RobertaModel, RobertaConfig, RobertaForMaskedLM
from model_center.model.config import RobertaConfig as myConfig

base_path = '/home/zhaoweilin/ModelCenter'

def convert_model(version: str):
    config: RobertaConfig = RobertaConfig.from_pretrained(version)

    num_layers = config.num_hidden_layers
    lmhead_bert = RobertaForMaskedLM.from_pretrained(version)
    dict = lmhead_bert.state_dict()
    new_dict = OrderedDict()

    new_dict['input_embedding.weight'] = dict['roberta.embeddings.word_embeddings.weight']
    new_dict['position_embedding.weight'] = dict['roberta.embeddings.position_embeddings.weight']
    new_dict['token_type_embedding.weight'] = dict['roberta.embeddings.token_type_embeddings.weight']

    for i in range(num_layers):
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.weight'] = (
            dict['roberta.embeddings.LayerNorm.weight'] if i == 0
            else dict['roberta.encoder.layer.' + str(i - 1) + '.output.LayerNorm.weight'])
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.bias'] = (
            dict['roberta.embeddings.LayerNorm.bias'] if i == 0
            else dict['roberta.encoder.layer.' + str(i - 1) + '.output.LayerNorm.bias'])
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_q.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.query.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_q.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.query.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_k.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.key.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_k.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.key.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_v.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.value.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_v.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.value.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.dense.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.dense.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.LayerNorm.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.LayerNorm.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.intermediate.dense.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.intermediate.dense.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.output.dense.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.output.dense.bias']

    new_dict['encoder.output_layernorm.weight'] = dict[
        'roberta.encoder.layer.' + str(num_layers - 1) + '.output.LayerNorm.weight']
    new_dict['encoder.output_layernorm.bias'] = dict[
        'roberta.encoder.layer.' + str(num_layers - 1) + '.output.LayerNorm.bias']

    new_dict['lm_head.dense.weight'] = dict['lm_head.dense.weight']
    new_dict['lm_head.dense.bias'] = dict['lm_head.dense.bias']
    new_dict['lm_head.layer_norm.weight'] = dict['lm_head.layer_norm.weight']
    new_dict['lm_head.layer_norm.bias'] = dict['lm_head.layer_norm.bias']
    new_dict['lm_head.decoder.weight'] = dict['lm_head.decoder.weight']
    new_dict['lm_head.decoder.bias'] = dict['lm_head.decoder.bias']

    roberta = RobertaModel.from_pretrained(version)
    dict = roberta.state_dict()
    new_dict['pooler.dense.weight'] = dict['pooler.dense.weight']
    new_dict['pooler.dense.bias'] = dict['pooler.dense.bias']

    torch.save(new_dict, os.path.join(base_path, 'configs', 'roberta', version, 'pytorch_model.pt'))


if __name__ == "__main__":
    convert_model("roberta-base")
    convert_model("roberta-large")
