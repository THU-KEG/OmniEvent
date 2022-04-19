import pdb
from typing import Tuple 
import torch 
import numpy as np
from sklearn.metrics import f1_score


def compute_span_F1(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    final_preds, final_labels = [], []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == -100:
                continue
            final_preds.append(predictions[i][j])
            final_labels.append(labels[i][j])
    pos_labels = list(set(final_labels))
    pos_labels.remove(0)
    micro_f1 = f1_score(final_labels, final_preds, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}
    

def compute_F1(eval_pred):
    logits, labels = eval_pred
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


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(softmax(logits), axis=-1)
    accuracy = (predictions == labels).sum() / labels.shape[0]
    return {"accuracy": accuracy}


def compute_accuracy_for_qa(eval_pred):
    logits, labels = eval_pred
    begin_logits, end_logits = logits[0], logits[1]
    begin_labels, end_labels = labels[0], labels[1]
    begin_predictions = np.argmax(softmax(begin_logits), axis=-1)
    end_predictions = np.argmax(softmax(end_logits), axis=-1)
    correct = (begin_predictions == begin_labels) * (end_predictions == end_labels)
    accuracy = correct.sum() / correct.shape[0]
    return {"accuracy": accuracy}

