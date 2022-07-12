import os
import json
import jsonlines
import numpy as np

from tqdm import tqdm


def get_sub_files(input_test_file, input_test_pred_file=None, sub_size=5000):
    test_data = list(jsonlines.open(input_test_file))
    sub_data_folder = '/'.join(input_test_file.split('/')[:-1]) + '/test_cache/'
    os.makedirs(sub_data_folder, exist_ok=True)
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


def predict_ed(trainer, tokenizer, data_class, data_args, data_file):
    dataset = data_class(data_args, tokenizer, data_file)
    logits, labels, metrics = trainer.predict(
        test_dataset=dataset,
        ignore_keys=["loss"]
    )
    return logits, labels, metrics, dataset


def predict_sub_ed(trainer, tokenizer, data_class, data_args, data_file):
    data_file_full = data_file
    data_file_list = get_sub_files(input_test_file=data_file_full,
                                   sub_size=data_args.split_infer_size)

    logits_list, labels_list = [], []
    for data_file in tqdm(data_file_list, desc='Split Evaluate'):
        logits, labels, metrics, _ = predict_ed(trainer, tokenizer, data_class, data_args, data_file)
        logits_list.append(logits)
        labels_list.append(labels)

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    metrics = trainer.compute_metrics(logits=logits, labels=labels,
                                      **{"tokenizer": tokenizer, "training_args": trainer.args})

    dataset = data_class(data_args, tokenizer, data_file_full)
    return logits, labels, metrics, dataset


def predict_eae(trainer, tokenizer, data_class, data_args, training_args):
    test_dataset = data_class(data_args, tokenizer, data_args.test_file, data_args.test_pred_file)
    training_args.data_for_evaluation = test_dataset.get_data_for_evaluation()
    logits, labels, metrics = trainer.predict(test_dataset=test_dataset, ignore_keys=["loss"])

    return logits, labels, metrics, test_dataset


def predict_sub_eae(trainer, tokenizer, data_class, data_args, training_args):
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

    metrics = trainer.compute_metrics(logits=logits, labels=labels,
                                      **{"tokenizer": tokenizer, "training_args": trainer.args})

    data_args.test_file = test_file_full
    data_args.test_pred_file = test_pred_file_full

    test_dataset = data_class(data_args, tokenizer, data_args.test_file, data_args.test_pred_file)
    return logits, labels, metrics, test_dataset
