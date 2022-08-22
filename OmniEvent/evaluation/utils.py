import os
import json
import logging
import shutil

import jsonlines
import numpy as np

from tqdm import tqdm
from .convert_format import get_ace2005_trigger_detection_sl, get_ace2005_trigger_detection_s2s
from ..input_engineering.seq2seq_processor import extract_argument

logger = logging.getLogger(__name__)


def dump_preds(trainer, tokenizer, data_class, output_dir, model_args, data_args, training_args, mode="train"):
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
    logger.info("{}-Dump Preds-{}{}".format("-"*25, mode, "-"*25))
    logger.info("Test file: {}, Metrics: {}, Split_Infer: {}".format(data_file, metrics, data_args.split_infer))

    preds = get_pred_s2s(logits, tokenizer) if model_args.paradigm == "seq2seq" else np.argmax(logits, axis=-1)

    if model_args.paradigm == "token_classification":
        pred_labels = [data_args.id2type[pred] for pred in preds]
    elif model_args.paradigm == "sequence_labeling":
        pred_labels = get_ace2005_trigger_detection_sl(preds, labels, data_file, data_args, dataset.is_overflow)
    elif model_args.paradigm == "seq2seq":
        pred_labels = get_ace2005_trigger_detection_s2s(preds, labels, data_file, data_args, None)
    else:
        raise NotImplementedError

    save_path = os.path.join(output_dir, "{}_preds.json".format(mode))

    json.dump(pred_labels, open(save_path, "w", encoding='utf-8'), ensure_ascii=False)
    logger.info("ED {} preds dumped to {}\n ED finished!".format(mode, save_path))


def get_pred_s2s(logits, tokenizer):
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)

    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()

    preds = list()
    for i, pred in enumerate(decoded_preds):
        pred = clean_str(pred)
        arguments = extract_argument(pred, i, "NA")
        tmp = dict()
        for arg in arguments:
            tmp[arg[-1]] = arg[-2]
        preds.append(tmp)

    return preds


def predict(trainer, tokenizer, data_class, data_args, data_file, training_args):
    if training_args.task_name == "ED":
        pred_func = predict_sub_ed if data_args.split_infer else predict_ed
        return pred_func(trainer, tokenizer, data_class, data_args, data_file)

    elif training_args.task_name == 'EAE':
        pred_func = predict_sub_eae if data_args.split_infer else predict_eae
        return pred_func(trainer, tokenizer, data_class, data_args, training_args)

    else:
        raise NotImplementedError


def get_sub_files(input_test_file, input_test_pred_file=None, sub_size=5000):
    test_data = list(jsonlines.open(input_test_file))
    sub_data_folder = '/'.join(input_test_file.split('/')[:-1]) + '/test_cache/'

    # clear the cache dir before split evaluate
    if os.path.isdir(sub_data_folder):
        shutil.rmtree(sub_data_folder)
        logger.info("Cleared Existing Cache Dir")

    os.makedirs(sub_data_folder, exist_ok=False)
    output_test_files = []

    if input_test_pred_file:
        pred_data = json.load(open(input_test_pred_file, encoding='utf-8'))
        sub_pred_folder = '/'.join(input_test_pred_file.split('/')[:-1]) + '/test_cache/'
        os.makedirs(sub_pred_folder, exist_ok=True)
        output_pred_files = []

    pred_start = 0
    for sub_id, i in enumerate(range(0, len(test_data), sub_size)):
        test_data_sub = test_data[i: i+sub_size]
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


def predict_ed(trainer,
               tokenizer: str,
               data_class,
               data_args,
               data_file: str):
    """Predicts the test set of the event detection task.

    Predicts the test set of the event detection task. The prediction of logits and labels, evaluation metrics' results,
    and the dataset would be returned.

    Args:
        trainer:
            The trainer for event detection.
        tokenizer (`str`):
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


def predict_sub_ed(trainer,
                   tokenizer: str,
                   data_class,
                   data_args,
                   data_file: str):
    """Predicts the test set of the event detection task of subfile datasets.

    Predicts the test set of the event detection task of a list of datasets. The prediction of logits and labels are
    conducted separately on each file, and the evaluation metrics' results are calculated after concatenating the
    predictions together. Finally, the prediction of logits and labels, evaluation metrics' results, and the dataset
    would be returned.

    Args:
        trainer:
            The trainer for event detection.
        tokenizer (`str`):
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


def predict_eae(trainer,
                tokenizer: str,
                data_class,
                data_args,
                training_args):
    """Predicts the test set of the event argument extraction task.

    Predicts the test set of the event argument extraction task. The prediction of logits and labels, evaluation
    metrics' results, and the dataset would be returned.

    Args:
        trainer:
            The trainer for event detection.
        tokenizer (`str`):
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


def predict_sub_eae(trainer,
                    tokenizer: str,
                    data_class,
                    data_args,
                    training_args):
    """Predicts the test set of the event detection task of subfile datasets.

    Predicts the test set of the event detection task of a list of datasets. The prediction of logits and labels are
    conducted separately on each file, and the evaluation metrics' results are calculated after concatenating the
    predictions together. Finally, the prediction of logits and labels, evaluation metrics' results, and the dataset
    would be returned.

    Args:
        trainer:
            The trainer for event detection.
        tokenizer (`str`):
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