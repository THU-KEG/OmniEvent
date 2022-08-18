# ModelCenter's Documentation

ModelCenter implements PLMs (Pretrained Language Models) based on [BMTrain](https://bmtrain.readthedocs.io/en/latest/index.html>) backend.

## Main Advantages:

- <span style="color:green;font-weight:bold">Low-Resource</span>
- <span style="color:red;font-weight:bold">Efficient</span>
- <span style="color:orange;font-weight:bold">Extendable</span>

```eval_rst
.. toctree::
   :maxdepth: 2
   :caption: GETTING STARTED

   notes/installation.md
   notes/quickstart.md
   notes/benchmark.md
   notes/write_model.md
   notes/pretrain_data.md

.. toctree::
   :maxdepth: 1
   :caption: Models

   model/bert.rst
   model/gpt2.rst
   model/gptj.rst
   model/t5.rst
   model/cpm1.rst
   model/cpm2.rst

.. toctree::
   :maxdepth: 2
   :caption: PACKAGE REFERENCE

   api/module.rst
   api/block.rst

.. toctree::
   :maxdepth: 2
   :caption: Advanced


Indices and tables
==================

* :ref:`genindex`

```