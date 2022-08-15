=======================
BERT
=======================

`Bert <https://arxiv.org/abs/1810.04805>`_

We currently support loading the following checkpoint via ``Bert.from_pretrained(identifier)``

- bert-base-cased
- bert-base-uncased
- bert-large-cased
- bert-large-uncased
- bert-base-chinese
- bert-base-multilingual-cased

BertConfig
------------------------------------
.. autoclass:: model_center.model.BertConfig
   :members:

BertModel
------------------------------------
.. autoclass:: model_center.model.Bert
   :members:

BertTokenizer
------------------------------------
.. class:: model_center.tokenizer.BertTokenizer

The current implementation is mainly an alias to BertTokenizer of `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`_.
we will change to our SAM implementation in the future, which will be a more efficient tokenizer.