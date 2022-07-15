import pdb
from typing import Tuple 
import torch 
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as span_f1_score
from seqeval.scheme import IOB2
from ..input_engineering.mrc_converter import make_preditions, compute_mrc_F1_cls
from .seq2seq_parser import extract_argument


def f1_score_overall(preds, labels):
    total_true = 0
    for pred in preds:
        if pred in labels:
            total_true += 1
    precision = total_true / len(preds)
    recall = total_true / len(labels)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1


# def compute_seq_F1(logits, labels, **kwargs):
#     tokenizer = kwargs["tokenizer"]
#     training_args = kwargs["training_args"]
#     decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)
#     assert len(decoded_preds) == len(training_args.data_for_evaluation["roles"])
#     def clean_str(x_str):
#         for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
#             x_str = x_str.replace(to_remove_token, '')
#         return x_str.strip()
#     pred_arguments = []
#     for i, pred in enumerate(decoded_preds):
#         pred_arguments.append(decode_arguments(clean_str(pred), training_args.data_for_evaluation["roles"][i], tokenizer))
#     golden_arguments = training_args.data_for_evaluation["golden_arguments"]
#     final_preds, final_labels = get_final_preds_labels(pred_arguments, golden_arguments)
#     precision, recall, micro_f1 = f1_score_overall(final_preds, final_labels)
#     # pdb.set_trace()
#     # pos_labels = list(set(final_labels))
#     # pos_labels.remove("NA")
#     # micro_f1 = f1_score(final_labels, final_preds, labels=pos_labels, average="micro")
#     pdb.set_trace()
#     return {"micro_f1": micro_f1*100}



def compute_seq_F1(logits, labels, **kwargs):
    tokenizer = kwargs["tokenizer"]
    training_args = kwargs["training_args"]
    pred_types = training_args.data_for_evaluation["pred_types"]
    true_types = training_args.data_for_evaluation["true_types"]
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    assert len(true_types) == len(decoded_labels)
    assert len(decoded_preds) == len(decoded_labels)
    pred_arguments, golden_arguments = [], []
    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        pred = clean_str(pred)
        label = clean_str(label)
        # if pred_types[i] != "NA":
        pred_arguments.extend(extract_argument(pred, i, pred_types[i]))
        golden_arguments.extend(extract_argument(label, i, true_types[i]))
    precision, recall, micro_f1 = f1_score_overall(pred_arguments, golden_arguments)
    return {"micro_f1": micro_f1*100}


# def compute_seq_F1(logits, labels, **kwargs):
#     tokenizer = kwargs["tokenizer"]
#     training_args = kwargs["training_args"]
#     decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
#     # convert to structured predictions
#     converter = training_args.seq2seq_converter
#     # Extract structured knowledge from text
#     decoded_preds = converter.extract_from_text(decoded_preds, training_args.true_types)
#     # decoded_labels = converter.extract_from_text(decoded_labels, training_args.true_types)
#     decoded_labels = training_args.golden_arguments
#     assert len(decoded_preds) == len(decoded_labels)
#     # metric 
#     final_labels, final_preds = converter.convert_to_final_list(decoded_labels,
#                                                                 decoded_preds, 
#                                                                 training_args.data_for_evaluation["true_types"],
#                                                                 training_args.data_for_evaluation["pred_types"])
#     pos_labels = list(training_args.id2role.values())
#     pos_labels.remove(training_args.id2role[0])
#     micro_f1 = f1_score(final_labels, final_preds, labels=pos_labels, average="micro") * 100.0
#     return {"micro_f1": micro_f1}


def select_start_position(preds, labels, merge=True):
    final_preds = []
    final_labels = []

    if merge:
        final_preds = preds[labels != -100].tolist()
        final_labels = labels[labels != -100].tolist()
    else:
        for i in range(labels.shape[0]):
            final_preds.append(preds[i][labels[i] != -100].tolist())
            final_labels.append(labels[i][labels[i] != -100].tolist())

    return final_preds, final_labels


def convert_to_names(instances, id2label):
    name_instances = []
    for instance in instances:
        name_instances.append([id2label[item] for item in instance])
    return name_instances


def compute_span_F1(logits, labels, **kwargs):
    if len(logits.shape) == 3:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits
    # convert id to name
    training_args = kwargs["training_args"]
    if training_args.task_name == "EAE":
        id2label = {id: role for role, id in training_args.role2id.items()}
    elif training_args.task_name == "ED":
        id2label = {id: role for role, id in training_args.type2id.items()}
    else:
        raise ValueError("No such task!")
    final_preds, final_labels = select_start_position(preds, labels, False)
    final_preds = convert_to_names(final_preds, id2label)
    final_labels = convert_to_names(final_labels, id2label)
    # if the type is wrongly predicted, set arguments NA
    if training_args.task_name == "EAE":
        pred_types = training_args.data_for_evaluation["pred_types"]
        true_types = training_args.data_for_evaluation["true_types"]
        assert len(pred_types) == len(true_types)
        assert len(pred_types) == len(final_labels)
        for i, (pred, true) in enumerate(zip(pred_types, true_types)):
            if pred != true:
                final_preds[i] = [id2label[0]] * len(final_preds[i]) # set to NA

    micro_f1 = span_f1_score(final_labels, final_preds, mode='strict', scheme=IOB2) * 100.0
    return {"micro_f1": micro_f1}
    

def compute_F1(logits, labels, **kwargs):
    predictions = np.argmax(logits, axis=-1)
    training_args = kwargs["training_args"]
    # if the type is wrongly predicted, set arguments NA
    if training_args.task_name == "EAE":
        pred_types = training_args.data_for_evaluation["pred_types"]
        true_types = training_args.data_for_evaluation["true_types"]
        assert len(pred_types) == len(true_types)
        assert len(pred_types) == len(predictions)
        for i, (pred, true) in enumerate(zip(pred_types, true_types)):
            if pred != true:
                predictions[i] = 0 # set to NA
        pos_labels = list(set(training_args.role2id.values()))
    else:
        pos_labels = list(set(training_args.type2id.values()))
    pos_labels.remove(0)
    micro_f1 = f1_score(labels, predictions, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}


def softmax(logits, dim=-1):
    logits = torch.tensor(logits)
    return torch.softmax(logits, dim=dim).numpy()


def compute_accuracy(logits, labels, **kwargs):
    predictions = np.argmax(softmax(logits), axis=-1)
    accuracy = (predictions == labels).sum() / labels.shape[0]
    return {"accuracy": accuracy}


def compute_mrc_F1(logits, labels, **kwargs):
    start_logits, end_logits = np.split(logits, 2, axis=-1)
    training_args = kwargs["training_args"]
    all_predictions, all_labels = make_preditions(start_logits, end_logits, kwargs["training_args"])
    micro_f1 = compute_mrc_F1_cls(all_predictions, all_labels)
    return {"micro_f1": micro_f1}


