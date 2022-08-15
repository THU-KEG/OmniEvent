# How to write a new model

## Model Implementation

We implement our models in `model_center/model`

We provided commonly used [modules](https://bmtrain.readthedocs.io/en/latest/api/module.html) in `model_center/layer`, such as `Linear`, `LayerNorm`, `Embedding`, 
which are implemented based on [bmtrain.DistributedParameter](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.DistributedParameter)
and [bmtrain.DistributedModule](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.DistributedModule), for distributed training support.

We have also implemented common ways of combining modules in `model_center/layer`, which are [block](https://bmtrain.readthedocs.io/en/latest/api/block.html).
For example, `SelfAttentionBlock` combines Layernorm, Attention, Add&Norm together.
Each blocks has diverse option, e.g., `FFNBlock` supports `gated_relu`, `relu`, `gated_gelu`, `gelu`; blocks support pre-layernorm and post-layernorm.

With the help of these commonly used modules we provided, a new model can be written easily without many exceptions. You can just add the model specific feature into the common structure.

A classic transformer is implemented in the following structure:

We use [bmtrain.CheckpointBlock](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.CheckpointBlock), and
[bmtrain.TransformerBlockList](https://bmtrain.readthedocs.io/en/latest/api/bmtrain.html#bmtrain.TransformerBlockList) to wrap our transformer blocks.
These reducd the GPU memory usage by a great amount without adding lots of computation time.
For more information, see [BMTrain's Quick Start](https://bmtrain.readthedocs.io/en/latest/notes/quickstart-zh.html)

```
T5(
  (input_embedding): Embedding()
  (position_bias_enc): RelativePositionEmbedding()
  (position_bias_dec): RelativePositionEmbedding()
  (encoder): Encoder(
    (layers): bmtrain.TransformerBlockList(
      (0): bmtrain.CheckpointBlock(
        TransformerBlock(
          (self_att): SelfAttentionBlock(
            (layernorm_before_attention): LayerNorm()
            (attention): Attention(
              (project_q): Linear()
              (project_k): Linear()
              (project_v): Linear()
              (attention_out): Linear()
            )
          )
          (ffn): FFNBlock(
            (layernorm_before_ffn): LayerNorm()
            (ffn): FeedForward(
              (w_in): DenseACT(
                (w): Linear()
                (act): ReLU()
              )
              (w_out): Linear()
            )
          )
        )
      )
      (1): bmtrain.CheckpointBlock()
      .
      .
      .
    )
    (output_layernorm): LayerNorm()
  )
  (decoder): Decoder(
    (layers): bmtrain.TransformerBlockList(
      (0): bmtrain.CheckpointBlock(
        (self_att): SelfAttentionBlock(
          (layernorm_before_attention): LayerNorm()
          (attention): Attention(
            (project_q): Linear()
            (project_k): Linear()
            (project_v): Linear()
            (attention_out): Linear()
          )
        )
        (cross_att): CrossAttentionBlock(
          (layernorm_before_attention): LayerNorm()
          (attention): Attention(
            (project_q): Linear()
            (project_k): Linear()
            (project_v): Linear()
            (attention_out): Linear()
          )
        )
        (ffn): FFNBlock(
          (layernorm_before_ffn): LayerNorm()
          (ffn): FeedForward(
            (w_in): DenseACT(
              (w): Linear()
              (act): ReLU()
            )
            (w_out): Linear()
          )
        )
      )
      (1): bmtrain.CheckpointBlock()
      .
      .
      .
    )
    (output_layernorm): LayerNorm()
  )
  (output_projection): Linear(
    (weight): bmtrain.DistributedParameter()
    (bias): bmtrain.DistributedParameter()
  )
)
```

## Model Config

We add model configs in `model_center/model/config`

By inheriting `model_center.config.Config`, config class can parse json files with `config.from_json_file(path)` method,
the parsed json file are then save to the config class and used by model by instantiating model with `model(config)`.