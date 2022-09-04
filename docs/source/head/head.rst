Classification Head
===================

.. code-block:: python

    from .classification import LinearHead, MRCHead
    from .crf import CRF

``get_head``
------------

.. code-block:: python

    def get_head(config):
        if config.head_type == "linear":
            return LinearHead(config)
        elif config.head_type == "mrc":
            return MRCHead(config)
        elif config.head_type == "crf":
            return CRF(config.num_labels, batch_first=True)
        elif config.head_type in ["none", "None"] or config.head_type is None:
            return None
        else:
            raise ValueError("Invalid head_type %s in config" % config.head_type)
