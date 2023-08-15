Evaluation Utils
================

.. code-block:: python

    import os
    import json
    import shutil
    import logging
    import jsonlines
    import numpy as np

    from tqdm import tqdm
    from pathlib import Path
    from typing import List, Dict, Union, Tuple
    from transformers import PreTrainedTokenizer

    from ..trainer import Trainer
    from ..trainer_seq2seq import Seq2SeqTrainer
    from ..arguments import DataArguments, ModelArguments, TrainingArguments
    from ..input_engineering.seq2seq_processor import extract_argument
    from ..input_engineering.base_processor import EDDataProcessor, EAEDataProcessor
    from ..input_engineering.mrc_converter import make_predictions, find_best_thresh

    from .convert_format import get_trigger_detection_sl, get_trigger_detection_s2s

    logger = logging.getLogger(__name__)

``dump_preds``
--------------

Save the Event Detection predictions for further use in the Event Argument Extraction task.

**Args:**

- ``trainer``: The trainer for event detection.
- ``tokenizer``: The tokenizer proposed for the tokenization process.
- ``data_class``: The processor of the input data.
- ``output_dir``: The file path to dump the event detection predictions.
- ``model_args``: The pre-defined arguments for model configuration.
- ``data_args``: The pre-defined arguments for data processing.
- ``training_args``: The pre-defined arguments for training event detection model.
- ``mode``: The mode of the prediction, can be 'train', 'valid' or 'test'.

.. code-block:: python

    def dump_preds(trainer: Union[Trainer, Seq2SeqTrainer],
                   tokenizer: PreTrainedTokenizer,
                   data_class: type,
                   output_dir: Union[str,Path],
                   model_args: ModelArguments,
                   data_args: DataArguments,
                   training_args: TrainingArguments,
                   mode: str = "train",
                   ) -> None:
        """Dump the Event Detection predictions for each token in the dataset.
        Save the Event Detection predictions for further use in the Event Argument Extraction task.
        Args:
            trainer:
                The trainer for event detection.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            data_class:
                The processor of the input data.
            output_dir (`str`):
                The file path to dump the event detection predictions.
            model_args (`ModelArguments`):
                The pre-defined arguments for model configuration.
            data_args (`DataArguments`):
                The pre-defined arguments for data processing.
            training_args (`TrainingArguments`):
                The pre-defined arguments for training event detection model.
            mode (`str`):
                The mode of the prediction, can be 'train', 'valid' or 'test'.
        Returns:
            None
        """
        if mode == "train":
            data_file = data_args.train_file
        elif mode == "valid":
            data_file = data_args.validation_file
        elif mode == "test":
            data_file = data_args.test_file
        else:
            raise NotImplementedError

        logits, labels, metrics, dataset = predict(trainer=trainer, tokenizer=tokenizer, data_class=data_class,
                                                   data_args=data_args, data_file=data_file,
                                                   training_args=training_args)
        logger.info("\n")
        logger.info("{}-Dump Preds-{}{}".format("-" * 25, mode, "-" * 25))
        logger.info("Test file: {}, Metrics: {}, Split_Infer: {}".format(data_file, metrics, data_args.split_infer))

        preds = get_pred_s2s(logits, tokenizer) if model_args.paradigm == "seq2seq" else np.argmax(logits, axis=-1)

        if model_args.paradigm == "token_classification":
            pred_labels = [data_args.id2type[pred] for pred in preds]
        elif model_args.paradigm == "sequence_labeling":
            pred_labels = get_trigger_detection_sl(preds, labels, data_file, data_args, dataset.is_overflow)
        elif model_args.paradigm == "seq2seq":
            pred_labels = get_trigger_detection_s2s(preds, labels, data_file, data_args, None)
        else:
            raise NotImplementedError

        save_path = os.path.join(output_dir, "{}_preds.json".format(mode))

        json.dump(pred_labels, open(save_path, "w", encoding='utf-8'), ensure_ascii=False)
        logger.info("ED {} preds dumped to {}\n ED finished!".format(mode, save_path))

``get_pred_s2s``
----------------

Converts Seq2Seq output logits to textual Event Type Prediction in Event Detection task,
or to textual Argument Role Prediction in Event Argument Extraction task.

**Args:**

- ``logits``: The decoded logits of the Seq2Seq model.
- ``tokenizer``: A string indicating the tokenizer proposed for the tokenization process.
- ``pred_types``: The event detection predictions, only used in Event Argument Extraction task.

**Returns:**

- ``preds``: The textual predictions of the Event Type or Argument Role. A list of tuple lists, in which each tuple is (argument, role) or (trigger, event_type)

.. code-block:: python

    def get_pred_s2s(logits: np.array,
                     tokenizer: PreTrainedTokenizer,
                     pred_types: List[str] = None,
                     ) -> List[List[Tuple[str, str]]]:
        """Convert Seq2Seq output logits to textual Event Type Prediction or Argument Role Prediction.
        Convert Seq2Seq output logits to textual Event Type Prediction in Event Detection task,
            or to textual Argument Role Prediction in Event Argument Extraction task.
        Args:
            logits (`np.array`):
                The decoded logits of the Seq2Seq model.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            pred_types (`List[str]`):
                The event detection predictions, only used in Event Argument Extraction task.
        Returns:
            preds (`List[List[Tuple[str, str]]]`):
                The textual predictions of the Event Type or Argument Role.
                A list of tuple lists, in which each tuple is (argument, role) or (trigger, event_type)
        """

        decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)

        def clean_str(x_str):
            for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
                x_str = x_str.replace(to_remove_token, '')
            return x_str.strip()

        preds = list()
        for i, pred in enumerate(decoded_preds):
            pred = clean_str(pred)
            pred_type = pred_types[i] if pred_types else "NA"
            arguments = extract_argument(pred, i, pred_type)
            tmp = list()
            for arg in arguments:
                tmp.append((arg[-1], arg[-2]))
            preds.append(tmp)

        return preds

``get_pred_mrc``
----------------

Converts MRC output logits to textual Event Type Prediction in Event Detection task,
or to textual Argument Role Prediction in Event Argument Extraction task.

**Args:**

- ``logits``: The logits output of the MRC model.
- ``training_args``: The event detection predictions, only used in Event Argument Extraction task.

**Returns:**

- ``preds``: The textual predictions of the Event Type or Argument Role. A list of tuple lists, in which each tuple is (argument, role) or (trigger, event_type)

.. code-block:: python

    def get_pred_mrc(logits: np.array,
                     training_args: TrainingArguments,
                     ) -> List[List[Tuple[str, str]]]:
        """Convert MRC output logits to textual Event Type Prediction or Argument Role Prediction.
        Convert MRC output logits to textual Event Type Prediction in Event Detection task,
            or to textual Argument Role Prediction in Event Argument Extraction task.
        Args:
            logits (`np.array`):
                The logits output of the MRC model.
            training_args (`TrainingArguments`):
                The event detection predictions, only used in Event Argument Extraction task.
        Returns:
            preds (`List[List[Tuple[str, str]]]`):
                The textual predictions of the Event Type or Argument Role.
                A list of tuple lists, in which each tuple is (argument, role) or (trigger, event_type)
        """

        start_logits, end_logits = np.split(logits, 2, axis=-1)
        all_preds, all_labels = make_predictions(start_logits, end_logits, training_args)

        all_preds = sorted(all_preds, key=lambda x: x[-2])
        best_na_thresh = find_best_thresh(all_preds, all_labels)
        logger.info("Best thresh founded. %.6f" % best_na_thresh)

        final_preds = []
        for argument in all_preds:
            if argument[-2] < best_na_thresh:
                final_preds.append(argument[:-2] + argument[-1:])  # no na_prob

        return final_preds

``predict``
-----------

Predicts the test set of the event detection task. The prediction of logits and labels, evaluation metrics' results,
and the dataset would be returned.

**Args:**

- ``trainer``: The trainer for event detection.
- ``tokenizer``: The tokenizer proposed for the tokenization process.
- ``data_class``: The processor of the input data.
- ``data_args``: The pre-defined arguments for data processing.
- ``data_file``: A string representing the file path of the dataset.
- ``training_args``: The pre-defined arguments for training.

**Returns:**

- ``logits``: An numpy array of integers containing the predictions from the model to be decoded.
- ``labels``: An numpy array of integers containing the actual labels obtained from the annotated dataset.
- ``metrics``: The evaluation metrics result based on the predictions and annotations.
- ``dataset``: An instance of the testing dataset.

.. code-block:: python

    def predict(trainer: Union[Trainer, Seq2SeqTrainer],
                tokenizer: PreTrainedTokenizer,
                data_class: type,
                data_args: DataArguments,
                data_file: str,
                training_args: TrainingArguments,
                ) -> Tuple[np.array, np.array, Dict, Union[EDDataProcessor, EAEDataProcessor]]:
        """Predicts the test set of the Event Detection task or Event Argument Extraction task.
        Predicts the test set of the event detection task. The prediction of logits and labels, evaluation metrics' results,
        and the dataset would be returned.
        Args:
            trainer:
                The trainer for event detection.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            data_class:
                The processor of the input data.
            data_args:
                The pre-defined arguments for data processing.
            data_file (`str`):
                A string representing the file path of the dataset.
            training_args (`TrainingArguments`):
                The pre-defined arguments for training.
        Returns:
            logits (`np.ndarray`):
                An numpy array of integers containing the predictions from the model to be decoded.
            labels: (`np.ndarray`):
                An numpy array of integers containing the actual labels obtained from the annotated dataset.
            metrics:
                The evaluation metrics result based on the predictions and annotations.
            dataset:
                An instance of the testing dataset.
        """

        if training_args.task_name == "ED":
            pred_func = predict_sub_ed if data_args.split_infer else predict_ed
            return pred_func(trainer, tokenizer, data_class, data_args, data_file)

        elif training_args.task_name == 'EAE':
            pred_func = predict_sub_eae if data_args.split_infer else predict_eae
            return pred_func(trainer, tokenizer, data_class, data_args, training_args)

        else:
            raise NotImplementedError

``get_sub_files``
-----------------

Splits a large data file into several small data files for evaluation.
Sometimes, the test data file can be too large to make prediction due to GPU memory constrain.
Therefore, we split the large file into several smaller ones and make predictions on each.

**Args:**

- ``input_test_file``: The path to the large data file that needs to split.
- ``input_test_pred_file``: The path to the Event Detection Predictions of the input_test_file. Only used in Event Argument Extraction task.
- ``sub_size``: The number of items contained each split file.

**Returns:**

- if ``input_test_pred_file`` is not ``None``: (Event Argument Extraction task)
    - ``output_test_files``, ``output_pred_files``: The lists of paths to the split files.
- else:
    - ``output_test_files``: The list of paths to the split files.

.. code-block:: python

    def get_sub_files(input_test_file: str,
                      input_test_pred_file: str = None,
                      sub_size: int = 5000,
                      ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """Split a large data file into several small data files for evaluation.
        Sometimes, the test data file can be too large to make prediction due to GPU memory constrain.
        Therefore, we split the large file into several smaller ones and make predictions on each.
        Args:
            input_test_file (`str`):
                The path to the large data file that needs to split.
            input_test_pred_file (`str`):
                The path to the Event Detection Predictions of the input_test_file.
                Only used in Event Argument Extraction task.
            sub_size (`int`):
                The number of items contained each split file.
        Returns:
            if input_test_pred_file is not None: (Event Argument Extraction task)
                output_test_files, output_pred_files:
                    The lists of paths to the split files.
            else:
                output_test_files:
                    The list of paths to the split files.
        """
        test_data = list(jsonlines.open(input_test_file))
        sub_data_folder = '/'.join(input_test_file.split('/')[:-1]) + '/test_cache/'

        # clear the cache dir before split evaluate
        if os.path.isdir(sub_data_folder):
            shutil.rmtree(sub_data_folder)
            logger.info("Cleared Existing Cache Dir")

        os.makedirs(sub_data_folder, exist_ok=False)
        output_test_files = []

        pred_data, sub_pred_folder = None, None
        output_pred_files = []
        if input_test_pred_file:
            pred_data = json.load(open(input_test_pred_file, encoding='utf-8'))
            sub_pred_folder = '/'.join(input_test_pred_file.split('/')[:-1]) + '/test_cache/'
            os.makedirs(sub_pred_folder, exist_ok=True)

        pred_start = 0
        for sub_id, i in enumerate(range(0, len(test_data), sub_size)):
            test_data_sub = test_data[i: i + sub_size]
            test_file_sub = sub_data_folder + 'sub-{}.json'.format(sub_id)

            with jsonlines.open(test_file_sub, 'w') as f:
                for data in test_data_sub:
                    jsonlines.Writer.write(f, data)

            output_test_files.append(test_file_sub)

            if input_test_pred_file:
                pred_end = pred_start + sum([len(d['candidates']) for d in test_data_sub])
                test_pred_sub = pred_data[pred_start: pred_end]
                pred_start = pred_end

                test_pred_file_sub = sub_pred_folder + 'sub-{}.json'.format(sub_id)

                with open(test_pred_file_sub, 'w', encoding='utf-8') as f:
                    json.dump(test_pred_sub, f, ensure_ascii=False)

                output_pred_files.append(test_pred_file_sub)

        if input_test_pred_file:
            return output_test_files, output_pred_files

        return output_test_files

``predict_ed``
--------------

Predicts the test set of the event detection task. The prediction of logits and labels, evaluation metrics' results,
and the dataset would be returned.

**Args:**

- ``trainer``: The trainer for event detection.
- ``tokenizer``: The tokenizer proposed for the tokenization process.
- ``data_class``: The processor of the input data.
- ``data_args``: The pre-defined arguments for data processing.
- ``data_file``: A string representing the file path of the dataset.

**Returns:**

- ``logits``: An numpy array of integers containing the predictions from the model to be decoded.
- ``labels``: An numpy array of integers containing the actual labels obtained from the annotated dataset.
- ``metrics``: The evaluation metrics result based on the predictions and annotations.
- ``dataset``: An instance of the testing dataset.

.. code-block:: python

    def predict_ed(trainer: Union[Trainer, Seq2SeqTrainer],
                   tokenizer: PreTrainedTokenizer,
                   data_class: type,
                   data_args,
                   data_file: str,
                   ) -> Tuple[np.array, np.array, Dict, EDDataProcessor]:
        """Predicts the test set of the event detection task.
        Predicts the test set of the event detection task. The prediction of logits and labels, evaluation metrics' results,
        and the dataset would be returned.
        Args:
            trainer:
                The trainer for event detection.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            data_class:
                The processor of the input data.
            data_args:
                The pre-defined arguments for data processing.
            data_file (`str`):
                A string representing the file path of the dataset.
        Returns:
            logits (`np.ndarray`):
                An numpy array of integers containing the predictions from the model to be decoded.
            labels: (`np.ndarray`):
                An numpy array of integers containing the actual labels obtained from the annotated dataset.
            metrics:
                The evaluation metrics result based on the predictions and annotations.
            dataset:
                An instance of the testing dataset.
        """
        dataset = data_class(data_args, tokenizer, data_file)
        logits, labels, metrics = trainer.predict(
            test_dataset=dataset,
            ignore_keys=["loss"]
        )
        return logits, labels, metrics, dataset

``predict_sub_ed``
------------------

Predicts the test set of the event detection task of a list of datasets. The prediction of logits and labels are
conducted separately on each file, and the evaluation metrics' results are calculated after concatenating the
predictions together. Finally, the prediction of logits and labels, evaluation metrics' results, and the dataset
would be returned.

Args:

- ``trainer``: The trainer for event detection.
- ``tokenizer``: The tokenizer proposed for the tokenization process.
- ``data_class``: The processor of the input data.
- ``data_args``: The pre-defined arguments for data processing.
- ``data_file``: A string representing the file path of the dataset.

**Returns:**

- ``logits``: An numpy array of integers containing the predictions from the model to be decoded.
- ``labels``: An numpy array of integers containing the actual labels obtained from the annotated dataset.
- ``metrics``: The evaluation metrics result based on the predictions and annotations.
- ``dataset``: An instance of the testing dataset.

.. code-block:: python

    def predict_sub_ed(trainer: Union[Trainer, Seq2SeqTrainer],
                       tokenizer: PreTrainedTokenizer,
                       data_class: type,
                       data_args: DataArguments,
                       data_file: str,
                       ) -> Tuple[np.array, np.array, Dict, EDDataProcessor]:
        """Predicts the test set of the event detection task of subfile datasets.
        Predicts the test set of the event detection task of a list of datasets. The prediction of logits and labels are
        conducted separately on each file, and the evaluation metrics' results are calculated after concatenating the
        predictions together. Finally, the prediction of logits and labels, evaluation metrics' results, and the dataset
        would be returned.
        Args:
            trainer:
                The trainer for event detection.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            data_class:
                The processor of the input data.
            data_args:
                The pre-defined arguments for data processing.
            data_file (`str`):
                A string representing the file path of the dataset.
        Returns:
            logits (`np.ndarray`):
                An numpy array of integers containing the predictions from the model to be decoded.
            labels: (`np.ndarray`):
                An numpy array of integers containing the actual labels obtained from the annotated dataset.
            metrics:
                The evaluation metrics result based on the predictions and annotations.
            dataset:
                An instance of the testing dataset.
        """
        data_file_full = data_file
        data_file_list = get_sub_files(input_test_file=data_file_full,
                                       sub_size=data_args.split_infer_size)

        logits_list, labels_list = [], []
        for data_file in tqdm(data_file_list, desc='Split Evaluate'):
            data_args.truncate_in_batch = False
            logits, labels, metrics, _ = predict_ed(trainer, tokenizer, data_class, data_args, data_file)
            logits_list.append(logits)
            labels_list.append(labels)

        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        metrics = trainer.compute_metrics(logits=logits, labels=labels,
                                          **{"tokenizer": tokenizer, "training_args": trainer.args})

        dataset = data_class(data_args, tokenizer, data_file_full)
        return logits, labels, metrics, dataset

``predict_eae``
---------------

Predicts the test set of the event argument extraction task. The prediction of logits and labels, evaluation
metrics' results, and the dataset would be returned.

Args:

- ``trainer``: The trainer for event detection.
- ``tokenizer``: A string indicating the tokenizer proposed for the tokenization process.
- ``data_class``: The processor of the input data.
- ``data_args``: The pre-defined arguments for data processing.
- ``training_args``: The pre-defined arguments for the training process.

**Returns:**

- ``logits``: An numpy array of integers containing the predictions from the model to be decoded.
- ``labels``: An numpy array of integers containing the actual labels obtained from the annotated dataset.
- ``metrics``: The evaluation metrics result based on the predictions and annotations.
- ``test_dataset``: An instance of the testing dataset.

.. code-block:: python

    def predict_eae(trainer: Union[Trainer, Seq2SeqTrainer],
                    tokenizer: PreTrainedTokenizer,
                    data_class: type,
                    data_args: DataArguments,
                    training_args: TrainingArguments,
                    ) -> Tuple[np.array, np.array, Dict, EAEDataProcessor]:
        """Predicts the test set of the event argument extraction task.
        Predicts the test set of the event argument extraction task. The prediction of logits and labels, evaluation
        metrics' results, and the dataset would be returned.
        Args:
            trainer:
                The trainer for event detection.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            data_class:
                The processor of the input data.
            data_args:
                The pre-defined arguments for data processing.
            training_args:
                The pre-defined arguments for the training process.
        Returns:
            logits (`np.ndarray`):
                An numpy array of integers containing the predictions from the model to be decoded.
            labels: (`np.ndarray`):
                An numpy array of integers containing the actual labels obtained from the annotated dataset.
            metrics:
                The evaluation metrics result based on the predictions and annotations.
            test_dataset:
                An instance of the testing dataset.
        """
        test_dataset = data_class(data_args, tokenizer, data_args.test_file, data_args.test_pred_file)
        training_args.data_for_evaluation = test_dataset.get_data_for_evaluation()
        logits, labels, metrics = trainer.predict(test_dataset=test_dataset, ignore_keys=["loss"])

        return logits, labels, metrics, test_dataset

``predict_sub_eae``
-------------------

Predicts the test set of the event detection task of a list of datasets. The prediction of logits and labels are
conducted separately on each file, and the evaluation metrics' results are calculated after concatenating the
predictions together. Finally, the prediction of logits and labels, evaluation metrics' results, and the dataset
would be returned.

**Args:**

- ``trainer``: The trainer for event detection.
- ``tokenizer``: The tokenizer proposed for the tokenization process.
- ``data_class``: The processor of the input data.
- ``data_args``: The pre-defined arguments for data processing.
- ``training_args``: The pre-defined arguments for the training process.

**Returns:**

- ``logits``: An numpy array of integers containing the predictions from the model to be decoded.
- ``labels``: An numpy array of integers containing the actual labels obtained from the annotated dataset.
- ``metrics``: The evaluation metrics result based on the predictions and annotations.
- ``test_dataset``: An instance of the testing dataset.

.. code-block:: python

    def predict_sub_eae(trainer: Union[Trainer, Seq2SeqTrainer],
                        tokenizer: PreTrainedTokenizer,
                        data_class: type,
                        data_args: DataArguments,
                        training_args: TrainingArguments,
                        ) -> Tuple[np.array, np.array, Dict, EDDataProcessor]:
        """Predicts the test set of the event detection task of subfile datasets.
        Predicts the test set of the event detection task of a list of datasets. The prediction of logits and labels are
        conducted separately on each file, and the evaluation metrics' results are calculated after concatenating the
        predictions together. Finally, the prediction of logits and labels, evaluation metrics' results, and the dataset
        would be returned.
        Args:
            trainer:
                The trainer for event detection.
            tokenizer (`PreTrainedTokenizer`):
                A string indicating the tokenizer proposed for the tokenization process.
            data_class:
                The processor of the input data.
            data_args:
                The pre-defined arguments for data processing.
            training_args:
                The pre-defined arguments for the training process.
        Returns:
            logits (`np.ndarray`):
                An numpy array of integers containing the predictions from the model to be decoded.
            labels: (`np.ndarray`):
                An numpy array of integers containing the actual labels obtained from the annotated dataset.
            metrics:
                The evaluation metrics result based on the predictions and annotations.
            test_dataset:
                An instance of the testing dataset.
        """
        test_file_full, test_pred_file_full = data_args.test_file, data_args.test_pred_file
        test_file_list, test_pred_file_list = get_sub_files(input_test_file=test_file_full,
                                                            input_test_pred_file=test_pred_file_full,
                                                            sub_size=data_args.split_infer_size)

        logits_list, labels_list = [], []
        for test_file, test_pred_file in tqdm(list(zip(test_file_list, test_pred_file_list)), desc='Split Evaluate'):
            data_args.test_file = test_file
            data_args.test_pred_file = test_pred_file

            logits, labels, metrics, _ = predict_eae(trainer, tokenizer, data_class, data_args, training_args)
            logits_list.append(logits)
            labels_list.append(labels)

        # TODO: concat operation is slow
        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        test_dataset_full = data_class(data_args, tokenizer, test_file_full, test_pred_file_full)
        training_args.data_for_evaluation = test_dataset_full.get_data_for_evaluation()

        metrics = trainer.compute_metrics(logits=logits, labels=labels,
                                          **{"tokenizer": tokenizer, "training_args": training_args})

        data_args.test_file = test_file_full
        data_args.test_pred_file = test_pred_file_full

        test_dataset = data_class(data_args, tokenizer, data_args.test_file, data_args.test_pred_file)
        return logits, labels, metrics, test_dataset
