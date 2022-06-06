import pdb
from typing import Tuple 
import torch 
import numpy as np
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as span_f1_score
from seqeval.scheme import IOB2


def postprocess_text(labels):
    parsed_labels = []
    for label in labels:
        label_per_instance = dict()
        for pair in label.split(";"):
            tt = pair.split(":")
            if len(tt) != 2:
                continue
            type, trigger = tt[0], tt[1]
            label_per_instance[trigger] = type 
        parsed_labels.append(label_per_instance)
    return parsed_labels


def compute_seq_F1(logits, labels, **kwargs):
    tokenizer = kwargs["tokenizer"]
    training_args = kwargs["training_args"]
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    # if training_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    converter = training_args.seq2seq_converter
    # Extract structured knowledge from text
    decoded_preds = converter.extract_from_text(decoded_preds, training_args.true_types)
    # decoded_labels = converter.extract_from_text(decoded_labels, training_args.true_types)
    decoded_labels = training_args.golden_arguments
    assert len(decoded_preds) == len(decoded_labels)
    # metric 
    final_labels, final_preds = converter.convert_to_final_list(decoded_labels,
                                                                decoded_preds, 
                                                                training_args.true_types,
                                                                training_args.pred_types)
    pos_labels = list(training_args.id2role.values())
    pos_labels.remove(training_args.id2role[0])
    micro_f1 = f1_score(final_labels, final_preds, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}

    # if "return_decoded_preds" in kwargs and kwargs["return_decoded_preds"]:
        # return decoded_preds



def select_start_position(preds, labels, merge=True):
    final_preds = [[] for _ in range(labels.shape[0])]
    final_labels = [[] for _ in range(labels.shape[0])]
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == -100:
                continue
            final_preds[i].append(preds[i][j])
            final_labels[i].append(labels[i][j])
    if merge:
        final_preds = [pred for preds in final_preds for pred in preds]
        final_labels = [label for labels in final_labels for label in labels]
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
    id2label = None 
    if training_args.task_name == "EAE":
        id2label = training_args.id2role 
    elif training_args.task_name == "ED":
        id2label = training_args.id2type 
    else:
        raise ValueError("No such task!")
    final_preds, final_labels = select_start_position(preds, labels, False)
    final_preds = convert_to_names(final_preds, id2label)
    final_labels = convert_to_names(final_labels, id2label)
    # if the type is wrongly predicted, set arguments NA
    if training_args.task_name == "EAE":
        pred_types = training_args.pred_types
        true_types = training_args.true_types 
        assert len(pred_types) == len(true_types)
        assert len(pred_types) == len(final_labels)
        for i, (pred, true) in enumerate(zip(pred_types, true_types)):
            if pred != true:
                final_preds[i] = [id2label[0]] * len(final_preds[i]) # set to NA

    micro_f1 = span_f1_score(final_labels, final_preds, mode='strict', scheme=IOB2) * 100.0
    return {"micro_f1": micro_f1}
    

def compute_F1(logits, labels, **kwargs):
    predictions = np.argmax(logits, axis=-1)
    pos_labels = list(set(labels.tolist()))
    pos_labels.remove(0)
    # convert id to name
    training_args = kwargs["training_args"]
    id2label = None 
    if training_args.task_name == "EAE":
        id2label = training_args.id2role
    elif training_args.task_name == "ED":
        id2label = training_args.id2type
    else:
        raise ValueError("No such task!")
    predictions = [id2label[p] for p in predictions]
    labels = [id2label[l] for l in labels]
    pos_labels = [id2label[pl] for pl in pos_labels]
    # if the type is wrongly predicted, set arguments NA
    if training_args.task_name == "EAE":
        pred_types = training_args.pred_types
        true_types = training_args.true_types 
        assert len(pred_types) == len(true_types)
        assert len(pred_types) == len(predictions)
        for i, (pred, true) in enumerate(zip(pred_types, true_types)):
            if pred != true:
                predictions[i] = id2label[0] # set to NA
    micro_f1 = f1_score(labels, predictions, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}


def softmax(logits, dim=-1):
    logits = torch.tensor(logits)
    return torch.softmax(logits, dim=dim).numpy()


def compute_accuracy(logits, labels, **kwargs):
    predictions = np.argmax(softmax(logits), axis=-1)
    accuracy = (predictions == labels).sum() / labels.shape[0]
    return {"accuracy": accuracy}


