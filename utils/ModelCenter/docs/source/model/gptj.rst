=======================
GPT-j
=======================

`GPTj <https://github.com/kingoflolz/mesh-transformer-jax>`_

We currently support loading the following checkpoint via ``GPTj.from_pretrained(identifier)``

- gptj-6b

GPTjConfig
------------------------------------
.. autoclass:: model_center.model.GPTjConfig
   :members:

GPTjModel
------------------------------------
.. autoclass:: model_center.model.GPTj
   :members:

GPTjTokenizer
------------------------------------
.. class:: model_center.tokenizer.GPTjTokenizer

The current implementation is mainly an alias to AutoTokenizer of `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`_.
we will change to our SAM implementation in the future, which will be a more efficient tokenizer.