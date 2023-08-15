Convert Format
==============

.. code-block:: python

    import json
    import logging
    import numpy as np

    from typing import List, Dict, Union, Tuple
    from sklearn.metrics import f1_score
    from .metric import select_start_position, compute_unified_micro_f1
    from ..input_engineering.input_utils import (
        get_left_and_right_pos,
        check_pred_len,
        get_ed_candidates,
        get_eae_candidates,
        get_event_preds,
        get_plain_label,
    )
    logger = logging.getLogger(__name__)

``get_pred_per_mention``
------------------------

Get the predicted event type or argument role for each mention via the predictions of different paradigms.
The predictions of Sequence Labeling, Seq2Seq, MRC paradigms are not aligned to each word. We need to convert the
paradigm-dependent predictions to word-level for the unified evaluation. This function is designed to get the
prediction for each single mention, given the paradigm-dependent predictions.

**Args:**

- ``pos_start``: The start position of the mention in the sequence of tokens.
- ``pos_end``: The end position of the mention in the sequence of tokens.
- ``preds``: The predictions of the sequence of tokens.
- ``id2label``: A dictionary that contains the mapping from id to textual label.
- ``label``: The ground truth label of the input mention.
- ``label2id``: A dictionary that contains the mapping from textual label to id.
- ``text``: The text of the input context.
- ``paradigm``: A string that indicates the paradigm.

**Returns:**

- A string which represents the predicted label.

.. code-block:: python

    def get_pred_per_mention(pos_start: int,
                             pos_end: int,
                             preds: List[Union[str, Tuple[str, str]]],
                             id2label: Dict[int, str] = None,
                             label: str = None,
                             label2id: Dict[str, int] = None,
                             text: str = None,
                             paradigm: str = "sl") -> str:
        """Get the predicted event type or argument role for each mention via the predictions of different paradigms.
        The predictions of Sequence Labeling, Seq2Seq, MRC paradigms are not aligned to each word. We need to convert the
        paradigm-dependent predictions to word-level for the unified evaluation. This function is designed to get the
        prediction for each single mention, given the paradigm-dependent predictions.
        Args:
            pos_start (`int`):
                The start position of the mention in the sequence of tokens.
            pos_end (`int`):
                The end position of the mention in the sequence of tokens.
            preds (`List[Union[str, Tuple[str, str]]]`):
                The predictions of the sequence of tokens.
            id2label (`Dict[int, str]`):
                A dictionary that contains the mapping from id to textual label.
            label (`str`):
                The ground truth label of the input mention.
            label2id (`Dict[str, int]`):
                A dictionary that contains the mapping from textual label to id.
            text (`str`):
                The text of the input context.
            paradigm (`str`):
                A string that indicates the paradigm.
        Returns:
            A string which represents the predicted label.
        """
        if paradigm == "sl":
            # sequence labeling paradigm
            if pos_start == pos_end or\
                    pos_end > len(preds) or \
                    id2label[int(preds[pos_start])] == "O" or \
                    id2label[int(preds[pos_start])].split("-")[0] != "B":
                return "NA"

            predictions = set()
            for pos in range(pos_start, pos_end):
                _pred = id2label[int(preds[pos])][2:]
                predictions.add(_pred)

            if len(predictions) > 1:
                return "NA"
            else:
                return list(predictions)[0]

        elif paradigm == "s2s":
            # seq2seq paradigm
            predictions = []
            word = text[pos_start: pos_end]
            for i, pred in enumerate(preds):
                if pred[0] == word:
                    if pred[1] in label2id:
                        pred_label = pred[1]
                        predictions.append(pred_label)
            if label in predictions:
                pred_label = label
            else:
                pred_label = predictions[0] if predictions else "NA"

            # remove the prediction that has been used for a specific mention.
            if (word, pred_label) in preds:
                preds.remove((word, pred_label))

            return pred_label

        elif paradigm == "mrc":
            # mrc paradigm
            predictions = []
            for pred in preds:
                if pred[1] == (pos_start, pos_end - 1):
                    pred_role = pred[0].split("_")[-1]
                    predictions.append(pred_role)

            if label in predictions:
                return label
            else:
                return predictions[0] if predictions else "NA"
        else:
            raise NotImplementedError

``get_trigger_detection_sl``
------------------------------------

Obtains the event detection prediction results of the ACE2005 dataset based on the sequence labeling paradigm,
predicting the labels and calculating the micro F1 score based on the predictions and labels.

**Args:**

- ``preds``: A list of strings indicating the predicted types of the instances.
- ``labels``: A list of strings indicating the actual labels of the instances.
- ``data_file``: A string indicating the path of the testing data file.
- ``data_args``: The pre-defined arguments for data processing.

**Returns:**

- ``results``: A list of strings indicating the prediction results of event triggers.

.. code-block:: python

    def get_trigger_detection_sl(preds: np.array,
                                         labels: np.array,
                                         data_file: str,
                                         data_args,
                                         is_overflow) -> List[str]:
        """Obtains the event detection prediction results of the ACE2005 dataset based on the sequence labeling paradigm.
        Obtains the event detection prediction results of the ACE2005 dataset based on the sequence labeling paradigm,
        predicting the labels and calculating the micro F1 score based on the predictions and labels.
        Args:
            preds (`np.array`):
                A list of strings indicating the predicted types of the instances.
            labels (`np.array`):
                A list of strings indicating the actual labels of the instances.
            data_file (`str`):
                A string indicating the path of the testing data file.
            data_args:
                The pre-defined arguments for data processing.
            is_overflow:
        Returns:
            results (`List[str]`):
                A list of strings indicating the prediction results of event triggers.
        """
        # get per-word predictions
        preds, labels = select_start_position(preds, labels, False)
        results = []
        label_names = []
        language = data_args.language

        with open(data_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                item = json.loads(line.strip())

                if not is_overflow[i]:
                    check_pred_len(pred=preds[i], item=item, language=language)

                candidates, label_names_per_item = get_ed_candidates(item=item)
                label_names.extend(label_names_per_item)

                # loop for converting
                for candidate in candidates:
                    left_pos, right_pos = get_left_and_right_pos(text=item["text"], trigger=candidate, language=language)
                    pred = get_pred_per_mention(left_pos, right_pos, preds[i], data_args.id2type)
                    results.append(pred)

        if "events" in item:
            micro_f1 = compute_unified_micro_f1(label_names=label_names, results=results)
            logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))

        return results

``get_argument_extraction_sl``
--------------------------------------

Obtains the event argument extraction prediction results of the ACE2005 dataset based on the sequence labeling
paradigm, predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
predictions and labels.

**Args:**

- ``preds``: A list of strings indicating the predicted types of the instances.
- ``labels``: A list of strings indicating the actual labels of the instances.
- ``data_file``: A string indicating the path of the testing data file.
- ``data_args``: The pre-defined arguments for data processing.

**Returns:**

- ``results``: A list of strings indicating the prediction results of event arguments.

.. code-block:: python

    def get_argument_extraction_sl(preds: np.array,
                                           labels: np.array,
                                           data_file: str,
                                           data_args,
                                           is_overflow) -> List[str]:
        """Obtains the event argument extraction results of the ACE2005 dataset based on the sequence labeling paradigm.
        Obtains the event argument extraction prediction results of the ACE2005 dataset based on the sequence labeling
        paradigm, predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
        predictions and labels.
        Args:
            preds (`np.array`):
                A list of strings indicating the predicted types of the instances.
            labels (`np.array`):
                A list of strings indicating the actual labels of the instances.
            data_file (`str`):
                A string indicating the path of the testing data file.
            data_args:
                The pre-defined arguments for data processing.
            is_overflow:
        Returns:
            results (`List[str]`):
                A list of strings indicating the prediction results of event arguments.
        """
        # evaluation mode
        eval_mode = data_args.eae_eval_mode
        language = data_args.language
        golden_trigger = data_args.golden_trigger

        # pred events
        event_preds = get_event_preds(pred_file=data_args.test_pred_file)

        # get per-word predictions
        preds, labels = select_start_position(preds, labels, False)
        results = []
        label_names = []
        with open(data_file, "r", encoding="utf-8") as f:
            trigger_idx = 0
            eae_instance_idx = 0
            lines = f.readlines()
            for line in lines:
                item = json.loads(line.strip())
                text = item["text"]
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        true_type = event["type"]
                        pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                        trigger_idx += 1

                        if eval_mode in ['default', 'loose']:
                            if pred_type == "NA":
                                continue

                        if not is_overflow[eae_instance_idx]:
                            check_pred_len(pred=preds[eae_instance_idx], item=item, language=language)

                        candidates, label_names_per_trigger = get_eae_candidates(item=item, trigger=trigger)
                        label_names.extend(label_names_per_trigger)

                        # loop for converting
                        for candi in candidates:
                            if true_type == pred_type:
                                # get word positions
                                left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candi, language=language)
                                # get predictions
                                pred = get_pred_per_mention(left_pos, right_pos, preds[eae_instance_idx], data_args.id2role)
                            else:
                                pred = "NA"
                            # record results
                            results.append(pred)
                        eae_instance_idx += 1

                # negative triggers
                for trigger in item["negative_triggers"]:
                    true_type = "NA"
                    pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                    trigger_idx += 1

                    if eval_mode in ['default', 'strict']:  # loose mode has no neg
                        if pred_type != "NA":
                            if not is_overflow[eae_instance_idx]:
                                check_pred_len(pred=preds[eae_instance_idx], item=item, language=language)

                            candidates, label_names_per_trigger = get_eae_candidates(item=item, trigger=trigger)
                            label_names.extend(label_names_per_trigger)

                            # loop for converting
                            for candi in candidates:
                                # get word positions
                                left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candi, language=language)
                                # get predictions
                                pred = get_pred_per_mention(left_pos, right_pos, preds[eae_instance_idx], data_args.id2role)
                                # record results
                                results.append(pred)

                            eae_instance_idx += 1

            assert len(preds) == eae_instance_idx

        pos_labels = list(set(label_names))
        pos_labels.remove("NA")
        micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0

        logger.info('Number of Instances: {}'.format(eae_instance_idx))
        logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))
        return results

``get_argument_extraction_mrc``
---------------------------------------

Obtains the event argument extraction prediction results of the ACE2005 dataset based on the MRC paradigm,
predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
predictions and labels.

**Args:**

- ``preds``: A list of strings indicating the predicted types of the instances.
- ``labels``: A list of strings indicating the actual labels of the instances.
- ``data_file``: A string indicating the path of the testing data file.
- ``data_args``: The pre-defined arguments for data processing.

**Returns:**

- ``results``: A list of strings indicating the prediction results of event arguments.

.. code-block:: python

    def get_argument_extraction_mrc(preds, labels, data_file, data_args, is_overflow):
        """Obtains the event argument extraction results of the ACE2005 dataset based on the MRC paradigm.
        Obtains the event argument extraction prediction results of the ACE2005 dataset based on the MRC paradigm,
        predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
        predictions and labels.
        Args:
            preds (`np.array`):
                A list of strings indicating the predicted types of the instances.
            labels (`np.array`):
                A list of strings indicating the actual labels of the instances.
            data_file (`str`):
                A string indicating the path of the testing data file.
            data_args:
                The pre-defined arguments for data processing.
            is_overflow:
        Returns:
            results (`List[str]`):
                A list of strings indicating the prediction results of event arguments.
        """

        # evaluation mode
        eval_mode = data_args.eae_eval_mode
        golden_trigger = data_args.golden_trigger
        language = data_args.language

        # pred events
        event_preds = get_event_preds(pred_file=data_args.test_pred_file)

        # get per-word predictions
        results = []
        all_labels = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            trigger_idx = 0
            eae_instance_idx = 0
            lines = f.readlines()
            for line in lines:
                item = json.loads(line.strip())
                text = item["text"]

                # preds per index
                preds_per_idx = []
                for pred in preds:
                    if pred[-1] == trigger_idx:
                        preds_per_idx.append(pred)

                for event in item["events"]:
                    for trigger in event["triggers"]:
                        true_type = event["type"]
                        pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                        trigger_idx += 1

                        if eval_mode in ['default', 'loose']:
                            if pred_type == "NA":
                                continue

                        # get candidates
                        candidates, labels_per_idx = get_eae_candidates(item, trigger)
                        all_labels.extend(labels_per_idx)

                        # loop for converting
                        for cid, candi in enumerate(candidates):
                            label = labels_per_idx[cid]
                            if pred_type == true_type:
                                # get word positions
                                left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candi, language=language)
                                # get predictions
                                pred_role = get_pred_per_mention(pos_start=left_pos, pos_end=right_pos, preds=preds_per_idx,
                                                                 label=label, paradigm='mrc')
                            else:
                                pred_role = "NA"
                            # record results
                            results.append(pred_role)
                        eae_instance_idx += 1

                # negative triggers
                for trigger in item["negative_triggers"]:
                    true_type = "NA"
                    pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                    trigger_idx += 1

                    if eval_mode in ['default', 'strict']:  # loose mode has no neg
                        if pred_type != "NA":
                            # get candidates
                            candidates, labels_per_idx = get_eae_candidates(item, trigger)
                            all_labels.extend(labels_per_idx)

                            # loop for converting
                            for candi in candidates:
                                label = "NA"
                                # get word positions
                                left_pos, right_pos = get_left_and_right_pos(text=text, trigger=candi, language=language)
                                # get predictions
                                pred_role = get_pred_per_mention(pos_start=left_pos, pos_end=right_pos, preds=preds_per_idx,
                                                                 label=label, paradigm='mrc')
                                # record results
                                results.append(pred_role)

                            eae_instance_idx += 1

        pos_labels = list(data_args.role2id.keys())
        pos_labels.remove("NA")
        micro_f1 = f1_score(all_labels, results, labels=pos_labels, average="micro") * 100.0

        logger.info('Number of Instances: {}'.format(eae_instance_idx))
        logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))
        return results

``get_trigger_detection_s2s``
-------------------------------------

Obtains the event detection prediction results of the ACE2005 dataset based on the Seq2Seq paradigm,
predicting the labels and calculating the micro F1 score based on the predictions and labels.

**Args:**

- ``preds``: A list of strings indicating the predicted types of the instances.
- ``labels``: A list of strings indicating the actual labels of the instances.
- ``data_file``: A string indicating the path of the testing data file.
- ``data_args``: The pre-defined arguments for data processing.

**Returns:**

- ``results``: A list of strings indicating the prediction results of event triggers.

.. code-block:: python

    def get_trigger_detection_s2s(preds, labels, data_file, data_args, is_overflow):
        """Obtains the event detection prediction results of the ACE2005 dataset based on the Seq2Seq paradigm.
        Obtains the event detection prediction results of the ACE2005 dataset based on the Seq2Seq paradigm,
        predicting the labels and calculating the micro F1 score based on the predictions and labels.
        Args:
            preds (`np.array`):
                A list of strings indicating the predicted types of the instances.
            labels (`np.array`):
                A list of strings indicating the actual labels of the instances.
            data_file (`str`):
                A string indicating the path of the testing data file.
            data_args:
                The pre-defined arguments for data processing.
            is_overflow:
        Returns:
            results (`List[str]`):
                A list of strings indicating the prediction results of event triggers.
        """
        # get per-word predictions
        results = []
        label_names = []
        with open(data_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                item = json.loads(line.strip())
                text = item["text"]
                preds_per_idx = preds[idx]

                candidates, labels_per_item = get_ed_candidates(item=item)
                for i, label in enumerate(labels_per_item):
                    labels_per_item[i] = get_plain_label(label)
                label_names.extend(labels_per_item)

                # loop for converting
                for cid, candidate in enumerate(candidates):
                    label = labels_per_item[cid]
                    # get word positions
                    left_pos, right_pos = candidate["position"]
                    # get predictions
                    pred_type = get_pred_per_mention(pos_start=left_pos, pos_end=right_pos, preds=preds_per_idx, text=text,
                                                     label=label, label2id=data_args.type2id, paradigm='s2s')
                    # record results
                    results.append(pred_type)

        if "events" in item:
            micro_f1 = compute_unified_micro_f1(label_names=label_names, results=results)
            logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))

        return results

``get_argument_extraction_s2s``
---------------------------------------

Obtains the event argument extraction prediction results of the ACE2005 dataset based on the Seq2Seq paradigm,
predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
predictions and labels.

**Args:**

- ``preds``: A list of strings indicating the predicted types of the instances.
- ``labels``: A list of strings indicating the actual labels of the instances.
- ``data_file``: A string indicating the path of the testing data file.
- ``data_args``: The pre-defined arguments for data processing.

**Returns:**

- ``results``: A list of strings indicating the prediction results of event arguments.

.. code-block:: python

    def get_argument_extraction_s2s(preds, labels, data_file, data_args, is_overflow):
        """Obtains the event argument extraction results of the ACE2005 dataset based on the Seq2Seq paradigm.
        Obtains the event argument extraction prediction results of the ACE2005 dataset based on the Seq2Seq paradigm,
        predicting the labels of entities and negative triggers and calculating the micro F1 score based on the
        predictions and labels.
        Args:
            preds (`np.array`):
                A list of strings indicating the predicted types of the instances.
            labels (`np.array`):
                A list of strings indicating the actual labels of the instances.
            data_file (`str`):
                A string indicating the path of the testing data file.
            data_args:
                The pre-defined arguments for data processing.
            is_overflow:
        Returns:
            results (`List[str]`):
                A list of strings indicating the prediction results of event arguments.
        """

        # evaluation mode
        eval_mode = data_args.eae_eval_mode
        golden_trigger = data_args.golden_trigger

        # pred events
        event_preds = get_event_preds(pred_file=data_args.test_pred_file)

        # get per-word predictions
        results = []
        all_labels = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            trigger_idx = 0
            eae_instance_idx = 0
            lines = f.readlines()
            for line in lines:
                item = json.loads(line.strip())
                text = item["text"]

                for event in item["events"]:
                    for trigger in event["triggers"]:
                        true_type = event["type"]
                        pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                        trigger_idx += 1

                        if eval_mode in ['default', 'loose']:
                            if pred_type == "NA":
                                continue

                        # preds per index
                        preds_per_idx = preds[eae_instance_idx]
                        # get candidates
                        candidates, labels_per_idx = get_eae_candidates(item, trigger)
                        for i, label in enumerate(labels_per_idx):
                            labels_per_idx[i] = get_plain_label(label)
                        all_labels.extend(labels_per_idx)

                        # loop for converting
                        for cid, candidate in enumerate(candidates):
                            label = labels_per_idx[cid]
                            if pred_type == true_type:
                                # get word positions
                                left_pos, right_pos = candidate["position"]
                                # get predictions
                                pred_role = get_pred_per_mention(pos_start=left_pos, pos_end=right_pos, preds=preds_per_idx,
                                                                 text=text, label=label, label2id=data_args.role2id,
                                                                 paradigm='s2s')
                            else:
                                pred_role = "NA"
                            # record results
                            results.append(pred_role)
                        eae_instance_idx += 1

                # negative triggers
                for trigger in item["negative_triggers"]:
                    true_type = "NA"
                    pred_type = true_type if golden_trigger or event_preds is None else event_preds[trigger_idx]
                    trigger_idx += 1

                    if eval_mode in ['default', 'strict']:  # loose mode has no neg
                        if pred_type != "NA":
                            # preds per index
                            preds_per_idx = preds[eae_instance_idx]

                            # get candidates
                            candidates, labels_per_idx = get_eae_candidates(item, trigger)
                            for i, label in enumerate(labels_per_idx):
                                labels_per_idx[i] = get_plain_label(label)
                            all_labels.extend(labels_per_idx)

                            # loop for converting
                            for cid, candidate in enumerate(candidates):
                                label = labels_per_idx[cid]
                                # get word positions
                                left_pos, right_pos = candidate["position"]
                                # get predictions
                                pred_role = get_pred_per_mention(pos_start=left_pos, pos_end=right_pos, preds=preds_per_idx,
                                                                 text=text, label=label, label2id=data_args.role2id,
                                                                 paradigm='s2s')
                                # record results
                                results.append(pred_role)

                            eae_instance_idx += 1

            assert len(preds) == eae_instance_idx

        pos_labels = list(data_args.role2id.keys())
        pos_labels.remove("NA")
        micro_f1 = f1_score(all_labels, results, labels=pos_labels, average="micro") * 100.0

        logger.info("Number of Instances: {}".format(eae_instance_idx))
        logger.info("{} test performance after converting: {}".format(data_args.dataset_name, micro_f1))
        return results
