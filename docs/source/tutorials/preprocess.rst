Convert the Dataset into Unified OmniEvent Format
=================================================

To simplify subsequent data loading and modeling, we provide pre-processing scripts for commonly-used Event Extraction
datasets. Users can download the dataset and convert it to the unified OmniEvent format by configuring the data path
defined in the ``run.sh`` file under the
`scripts/data_preprocessing <https://github.com/THU-KEG/OmniEvent/tree/main/scripts>`_ folder with the same name as
the dataset.

Unified OmniEvent Format
------------------------

A unified OmniEvent dataset is a `JSON Line <https://jsonlines.org/>`_ file with the extension `.unified.jsonl` (such
as, ``train.unified.jsonl``, ``valid.unified.jsonl``, and ``test.unified.jsonl``), which is a convenient format for
storing structured data that enables processing one record, in one line, at a time. Taking a record from TAC KBP 2016 as
an example, a piece of data in the unified OmniEvent format could be demonstrated as follows:

.. code-block:: json

    {
        "id": "NYT_ENG_20130910.0002-6",
        "text": "In 1997 , Chun was sentenced to life in prison and Roh to 17 years .",
        "events": [{
            "type": "sentence",
            "triggers": [{
                "id": "em-2342",
                "trigger_word": "sentenced",
                "position": [19, 28],
                "arguments": [{
                    "role": "defendant",
                    "mentions": [{
                        "id": "m-291",
                        "mention": "Chun",
                        "position": [10, 14]}]}, ... ]}, ... ]} ... ],
        "negative_triggers": [{
            "id": 0,
            "trigger_word": "In",
            "position": [0, 2]}, ... ],
        "entities":  [{
            "type": "PER",
            "mentions": [{
                "id": "m-291",
                "mention": "Chun",
                "position": [10, 14]}, ... ]}, ... ]}

Supported Datasets
------------------

The pre-processing scripts support almost all commonly-used Event Extraction datasets, so as to minimize the data
conversion difficulties. Additional pre-processing scripts are still being developed, and you can submit datasets for
which you wish us to complete in "`Pull requests <https://github.com/THU-KEG/OmniEvent/pulls>`_". Currently, we have
developed pre-processing scripts for the following datasets:

- **ACE2005:** ACE2005-EN, ACE2005-DyGIE, ACE2005-OneIE, ACE2005-ZH
- **DuEE:** DuEE1.0, DuEE-fin
- **ERE:** LDC2015E29, LDC2015E68, LDC2015E78
- **FewFC**
- **TAC KBP:** TAC KBP 2014, TAC KBP 2015, TAC KBP 2016, TAC KBP 2017
- **LEVEN**
- **MAVEN**

Dataset Conversion
------------------

Step 1: Download the Dataset
````````````````````````````

The first step of data conversion is to download the proposed dataset from its corresponding website. For example, for
the DuEE 1.0 dataset, it could be downloaded from `here <https://www.luge.ai/#/luge/dataDetail?id=6>`_.

Step 2: Configure the Dataset Path
``````````````````````````````````

After downloading the dataset from the Internet, the ``run.sh`` file under the folder with the same name as the dataset
should be configured. For example, for the DuEE 1.0 dataset, the ``run.sh`` file under the path
`scripts/data_preprocessing/duee <https://github.com/THU-KEG/OmniEvent/tree/main/scripts/data_processing/duee>`_
should be configured, in which the ``data_dir`` path should be the same as the path of placing the downloaded dataset,
you can also modify the path of the processed dataset by configuring the ``save_dir`` path:

.. code-block:: shell

    python duee.py \
        --data_dir ../../../data/original/DuEE1.0 \
        --save_dir ../../../data/processed/DuEE1.0

Step 3: Execute the ``run.sh`` File
`````````````````````````````````

After downloading the dataset and configuring the corresponding ``run.sh`` file, finally, the dataset could finally be
converted to the unified OmniEvent format by executing the configured ``run.sh`` file. For example, for the DuEE1.0
dataset, we could execute the ``run.sh`` file as follows:

.. code-block:: shell

    bash run.sh
