Tuning Large PLMs for Event Extraction
======================================

We provide an example script for tuning large pre-trained language models (PLMs) on event extraction tasks. We use BMTrain as the distributed training engine. BMTrain is an efficient large model training toolkit, see `BMTrain <https://github.com/OpenBMB/BMTrain>`_ and `ModelCenter <https://github.com/OpenBMB/ModelCenter>`_ for more details. We adapt the code of ModelCenter for event extraction and place the code in `OmniEvent/utils`.

Setup
-----

Install the code in ``OmniEvent/utils/ModelCenter``:

.. code-block:: shell

    cd utils/ModelCenter
    pip install .

Easy Start
----------

Run ``bash train.sh``  to train MT5-xxl. You can modify the config and the important hyper-parameters are as follows:

.. code-block:: python

    NNODES # number of nodes
    GPUS_PER_NODE # gpus use on one node
    model-config # We only support T5 and MT5

The original ModelCenter repo doesn't support inference method (i.e. ``generate``) for decoder PLMs. We provide
``beam_search.py`` for inference.






