import pdb
from typing import Tuple 
import torch 
import numpy as np
from sklearn.metrics import f1_score


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
    if training_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = postprocess_text(decoded_preds)
    decoded_labels = postprocess_text(decoded_labels)

    assert len(decoded_preds) == len(decoded_labels)

    if "return_decoded_preds" in kwargs and kwargs["return_decoded_preds"]:
        return decoded_preds
    
    correct, total = 0, 0
    for i in range(len(decoded_preds)):
        pred = decoded_preds[i]
        label = decoded_labels[i]
        for key in pred:
            if key in label and pred[key] == label[key]:
                correct += 1
        total += len(label)
    total = 1e-10 if total == 0 else total 
    return dict(
        accuracy=correct/total,
        correct=correct,
        total=total
    )


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


def compute_span_F1(logits, labels, **kwargs):
    if len(logits.shape) == 3:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits
    final_preds, final_labels = select_start_position(preds, labels)
    pos_labels = list(set(final_labels))
    pos_labels.remove(0)
    micro_f1 = f1_score(final_labels, final_preds, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}
    

def compute_F1(logits, labels, **kwargs):
    predictions = np.argmax(logits, axis=-1)
    pos_labels = list(set(labels.tolist()))
    # pdb.set_trace()
    # remove negative label
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


