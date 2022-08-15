import os
import json
from sklearn.metrics import f1_score

from .metric import select_start_position
from .dump_result import get_pred_per_mention


def get_ace2005_trigger_detection_sl(preds, labels, data_file, data_args, is_overflow):
    # get per-word predictions
    preds, labels = select_start_position(preds, labels, False)
    results = []
    label_names = []
    with open(data_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line.strip())
            if not is_overflow[i]:
                if data_args.language == "English":
                    assert len(preds[i]) == len(item["text"].split())
                elif data_args.language == "Chinese":
                    assert len(preds[i]) == len("".join(item["text"].split()))  # remove space token
                else:
                    raise NotImplementedError

            candidates = []
            if "events" in item:
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        label_names.append(event["type"])
                        candidates.append(trigger)
                for neg_trigger in item["negative_triggers"]:
                    label_names.append("NA")
                    candidates.append(neg_trigger)
            else:
                candidates = item["candidates"]

            # loop for converting 
            for candidate in candidates:
                # get word positions
                char_pos = candidate["position"]
                if data_args.language == "English":
                    word_pos_start = len(item["text"][:char_pos[0]].split())
                    word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                elif data_args.language == "Chinese":
                    word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                    word_pos_end = len("".join(item["text"][:char_pos[1]].split()))
                else:
                    raise NotImplementedError
                # get predictions
                pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[i], data_args.id2type)
                # record results
                results.append(pred)

    if "events" in item:
        pos_labels = list(set(label_names))
        pos_labels.remove("NA")
        micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0
        print("After converting, the micro_f1 is %.4f" % micro_f1)

    return results


def get_ace2005_argument_extraction_sl(preds, labels, data_file, data_args, is_overflow):
    # evaluation mode
    eval_mode = data_args.eae_eval_mode

    # pred events
    pred_file = data_args.test_pred_file
    if pred_file is not None and os.path.exists(pred_file):
        event_preds = json.load(open(pred_file))
    else:
        event_preds = None
        print('Loading ED test_pred file failed, using golden triggers for EAE evaluation')

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
            for event in item["events"]:
                for trigger in event["triggers"]:
                    true_type = event["type"]
                    if data_args.golden_trigger or event_preds is None:
                        pred_type = true_type
                    else:
                        pred_type = event_preds[trigger_idx]

                    trigger_idx += 1

                    if eval_mode in ['default', 'loose']:
                        if pred_type == "NA":
                            continue

                    if not is_overflow[eae_instance_idx]:
                        if data_args.language == "English":
                            assert len(preds[eae_instance_idx]) == len(item["text"].split())
                        elif data_args.language == "Chinese":
                            assert len(preds[eae_instance_idx]) == len("".join(item["text"].split()))  # remove space
                        else:
                            raise NotImplementedError

                    candidates = []
                    positive_mentions = set()
                    positive_offsets = []
                    for argument in trigger["arguments"]:
                        for mention in argument["mentions"]:
                            label_names.append(argument["role"])
                            candidates.append(mention)
                            positive_mentions.add(mention["mention_id"])
                            positive_offsets.append(mention["position"])

                    if "entities" in item:
                        for entity in item["entities"]:
                            # check whether the entity is an argument
                            is_argument = False
                            for mention in entity["mentions"]:
                                if mention["mention_id"] in positive_mentions:
                                    is_argument = True
                                    break
                            if is_argument:
                                continue
                            # negative arguments
                            for mention in entity["mentions"]:
                                label_names.append("NA")
                                candidates.append(mention)
                    else:
                        for neg in item["negative_triggers"]:
                            is_argument = False
                            neg_set = set(range(neg["position"][0], neg["position"][1]))
                            for pos_offset in positive_offsets:
                                pos_set = set(range(pos_offset[0], pos_offset[1]))
                                if not pos_set.isdisjoint(neg_set):
                                    is_argument = True
                                    break
                            if is_argument:
                                continue
                            label_names.append("NA")
                            candidates.append(neg)

                    # loop for converting
                    for candidate in candidates:
                        # get word positions
                        char_pos = candidate["position"]
                        if data_args.language == "English":
                            word_pos_start = len(item["text"][:char_pos[0]].split())
                            word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                        elif data_args.language == "Chinese":
                            word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                            word_pos_end = len("".join(item["text"][:char_pos[1]].split()))
                        else:
                            raise NotImplementedError
                        # get predictions
                        pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[eae_instance_idx], data_args.id2role)
                        # record results
                        results.append(pred)
                    eae_instance_idx += 1

            # negative triggers
            for trigger in item["negative_triggers"]:
                true_type = "NA"
                if data_args.golden_trigger or event_preds is None:
                    pred_type = true_type
                else:
                    pred_type = event_preds[trigger_idx]
                trigger_idx += 1

                if eval_mode in ['default', 'strict']:  # loose mode has no neg
                    if pred_type != "NA":
                        if not is_overflow[eae_instance_idx]:
                            if data_args.language == "English":
                                assert len(preds[eae_instance_idx]) == len(item["text"].split())
                            elif data_args.language == "Chinese":
                                assert len(preds[eae_instance_idx]) == len(
                                    "".join(item["text"].split()))  # remove space
                            else:
                                raise NotImplementedError

                        candidates = []
                        for neg in item["negative_triggers"]:
                            label_names.append("NA")
                            candidates.append(neg)

                        # loop for converting
                        for candidate in candidates:
                            # get word positions
                            char_pos = candidate["position"]
                            if data_args.language == "English":
                                word_pos_start = len(item["text"][:char_pos[0]].split())
                                word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                            elif data_args.language == "Chinese":
                                word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                                word_pos_end = len("".join(item["text"][:char_pos[1]].split()))
                            else:
                                raise NotImplementedError
                            # get predictions
                            pred = get_pred_per_mention(word_pos_start, word_pos_end, preds[eae_instance_idx], data_args.id2role)
                            # record results
                            results.append(pred)

                        eae_instance_idx += 1

        assert len(preds) == eae_instance_idx
        
    pos_labels = list(set(label_names))
    pos_labels.remove("NA")
    micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0

    print('Evaluate Mode: {}, Number of Instances: {}'.format(eval_mode, eae_instance_idx))
    print("After converting, the micro_f1 is %.4f" % micro_f1)
    return results


def get_eae_candidates(item, trigger):
    candidates = []
    positive_mentions = set()
    positive_offsets = []
    label_names = []
    for argument in trigger["arguments"]:
        for mention in argument["mentions"]:
            label_names.append(argument["role"])
            candidates.append(mention)
            positive_mentions.add(mention["mention_id"])
            positive_offsets.append(mention["position"])

    if "entities" in item:
        for entity in item["entities"]:
            # check whether the entity is an argument
            is_argument = False
            for mention in entity["mentions"]:
                if mention["mention_id"] in positive_mentions:
                    is_argument = True
                    break
            if is_argument:
                continue
            # negative arguments
            for mention in entity["mentions"]:
                label_names.append("NA")
                candidates.append(mention)
    else:
        for neg in item["negative_triggers"]:
            is_argument = False
            neg_set = set(range(neg["position"][0], neg["position"][1]))
            for pos_offset in positive_offsets:
                pos_set = set(range(pos_offset[0], pos_offset[1]))
                if not pos_set.isdisjoint(neg_set):
                    is_argument = True
                    break
            if is_argument:
                continue
            label_names.append("NA")
            candidates.append(neg)

    return candidates, label_names


def get_ace2005_argument_extraction_mrc(preds, data_args):
    # evaluation mode
    eval_mode = data_args.eae_eval_mode

    # pred events
    pred_file = data_args.test_pred_file
    if pred_file is not None and os.path.exists(pred_file):
        event_preds = json.load(open(pred_file))
    else:
        event_preds = None
        print('Loading ED test_pred file failed, using golden triggers for EAE evaluation')

    # get per-word predictions
    results = []
    all_labels = []
    with open(data_args.test_file, "r", encoding="utf-8") as f:
        trigger_idx = 0
        eae_instance_idx = 0
        lines = f.readlines()
        for line in enumerate(lines):
            item = json.loads(line.strip())
            # preds per index 
            preds_per_idx = []
            for pred in preds:
                if pred[-1] == trigger_idx:
                    preds_per_idx.append(pred)

            for event in item["events"]:
                for trigger in event["triggers"]:
                    true_type = event["type"]
                    if data_args.golden_trigger or event_preds is None:
                        pred_type = true_type
                    else:
                        pred_type = event_preds[trigger_idx]

                    trigger_idx += 1

                    if eval_mode in ['default', 'loose']:
                        if pred_type == "NA":
                            continue

                    # get candidates 
                    candidates, labels_per_idx = get_eae_candidates(item, trigger)
                    all_labels.extend(labels_per_idx)

                    # loop for converting
                    for candidate in candidates:
                        # get word positions
                        char_pos = candidate["position"]
                        if data_args.language == "English":
                            word_pos_start = len(item["text"][:char_pos[0]].split())
                            word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split()) - 1
                        elif data_args.language == "Chinese":
                            word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                            word_pos_end = len("".join(item["text"][:char_pos[1]].split())) - 1
                        else:
                            raise NotImplementedError
                        # get predictions
                        pred_type = "NA"
                        for pred in preds_per_idx:
                            if pred[1] == (word_pos_start, word_pos_end):
                                pred_type = pred[0].split("_")[0]
                        # record results
                        results.append(pred)
                    eae_instance_idx += 1

            # negative triggers
            for trigger in item["negative_triggers"]:
                true_type = "NA"
                if data_args.golden_trigger or event_preds is None:
                    pred_type = true_type
                else:
                    pred_type = event_preds[trigger_idx]
                trigger_idx += 1

                if eval_mode in ['default', 'strict']:  # loose mode has no neg
                    if pred_type != "NA":
                        candidates = []
                        for neg in item["negative_triggers"]:
                            candidates.append(neg)

                        # loop for converting
                        for candidate in candidates:
                            # get word positions
                            char_pos = candidate["position"]
                            if data_args.language == "English":
                                word_pos_start = len(item["text"][:char_pos[0]].split())
                                word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split()) - 1
                            elif data_args.language == "Chinese":
                                word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                                word_pos_end = len("".join(item["text"][:char_pos[1]].split())) - 1
                            else:
                                raise NotImplementedError
                            # get predictions
                            pred_type = "NA"
                            for pred in preds_per_idx:
                                if pred[1] == (word_pos_start, word_pos_end):
                                    pred_type = pred[0].split("_")[0]
                            # record results
                            results.append(pred_type)

                        eae_instance_idx += 1

        assert len(preds) == eae_instance_idx
        
    pos_labels = list(data_args.role2id.keys())
    pos_labels.remove("NA")
    micro_f1 = f1_score(all_labels, results, labels=pos_labels, average="micro") * 100.0

    print('Evaluate Mode: {}, Number of Instances: {}'.format(eval_mode, eae_instance_idx))
    print("After converting, the micro_f1 is %.4f" % micro_f1)
    return results


def get_ace2005_trigger_detection_s2s(preds, data_file, data_args):
    # get per-word predictions
    results = []
    label_names = []
    with open(data_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            # preds per index 
            preds_per_idx = []
            for pred in preds:
                if pred[0] == idx:
                    preds_per_idx.append(pred)

            candidates = []
            if "events" in item:
                for event in item["events"]:
                    for trigger in event["triggers"]:
                        label_names.append(event["type"])
                        candidates.append(trigger)
                for neg_trigger in item["negative_triggers"]:
                    label_names.append("NA")
                    candidates.append(neg_trigger)
            else:
                candidates = item["candidates"]

            # loop for converting 
            for candidate in candidates:
                # get word positions
                char_pos = candidate["position"]
                if data_args.language == "English":
                    word_pos_start = len(item["text"][:char_pos[0]].split())
                    word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                elif data_args.language == "Chinese":
                    word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                    word_pos_end = len("".join(item["text"][:char_pos[1]].split()))
                else:
                    raise NotImplementedError
                # get predictions
                candidate_mention = item["text"][char_pos[0]:char_pos[1]]
                pred_type = "NA"
                for pred in preds_per_idx:
                    if pred[-1] == candidate_mention:
                        pred_type = pred[-2]
                # record results
                results.append(pred_type)

    if "events" in item:
        pos_labels = list(set(label_names))
        pos_labels.remove("NA")
        micro_f1 = f1_score(label_names, results, labels=pos_labels, average="micro") * 100.0
        print("After converting, the micro_f1 is %.4f" % micro_f1)

    return results


def get_ace2005_argument_extraction_s2s(preds, data_args):
    # evaluation mode
    eval_mode = data_args.eae_eval_mode

    # pred events
    pred_file = data_args.test_pred_file
    if pred_file is not None and os.path.exists(pred_file):
        event_preds = json.load(open(pred_file))
    else:
        event_preds = None
        print('Loading ED test_pred file failed, using golden triggers for EAE evaluation')

    # get per-word predictions
    results = []
    all_labels = []
    with open(data_args.test_file, "r", encoding="utf-8") as f:
        trigger_idx = 0
        eae_instance_idx = 0
        lines = f.readlines()
        for line in enumerate(lines):
            item = json.loads(line.strip())
            # preds per index 
            preds_per_idx = []
            for pred in preds:
                if pred[-1] == trigger_idx:
                    preds_per_idx.append(pred)

            for event in item["events"]:
                for trigger in event["triggers"]:
                    true_type = event["type"]
                    if data_args.golden_trigger or event_preds is None:
                        pred_type = true_type
                    else:
                        pred_type = event_preds[trigger_idx]

                    trigger_idx += 1

                    if eval_mode in ['default', 'loose']:
                        if pred_type == "NA":
                            continue

                    # get candidates 
                    candidates, labels_per_idx = get_eae_candidates(item, trigger)
                    all_labels.extend(labels_per_idx)

                    # loop for converting
                    for candidate in candidates:
                        # get word positions
                        char_pos = candidate["position"]
                        if data_args.language == "English":
                            word_pos_start = len(item["text"][:char_pos[0]].split())
                            word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split())
                        elif data_args.language == "Chinese":
                            word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                            word_pos_end = len("".join(item["text"][:char_pos[1]].split()))
                        else:
                            raise NotImplementedError
                        # get predictions
                        candidate_mention = item["text"][char_pos[0]:char_pos[1]]
                        pred_type = "NA"
                        for pred in preds_per_idx:
                            if pred[-1] == candidate_mention:
                                pred_type = pred[-2]
                        # record results
                        results.append(pred_type)
                    eae_instance_idx += 1

            # negative triggers
            for trigger in item["negative_triggers"]:
                true_type = "NA"
                if data_args.golden_trigger or event_preds is None:
                    pred_type = true_type
                else:
                    pred_type = event_preds[trigger_idx]
                trigger_idx += 1

                if eval_mode in ['default', 'strict']:  # loose mode has no neg
                    if pred_type != "NA":
                        candidates = []
                        for neg in item["negative_triggers"]:
                            candidates.append(neg)

                        # loop for converting
                        for candidate in candidates:
                            # get word positions
                            char_pos = candidate["position"]
                            if data_args.language == "English":
                                word_pos_start = len(item["text"][:char_pos[0]].split())
                                word_pos_end = word_pos_start + len(item["text"][char_pos[0]:char_pos[1]].split()) - 1
                            elif data_args.language == "Chinese":
                                word_pos_start = len("".join(item["text"][:char_pos[0]].split()))
                                word_pos_end = len("".join(item["text"][:char_pos[1]].split())) - 1
                            else:
                                raise NotImplementedError
                            # get predictions
                            candidate_mention = item["text"][char_pos[0]:char_pos[1]]
                            pred_type = "NA"
                            for pred in preds_per_idx:
                                if pred[-1] == candidate_mention:
                                    pred_type = pred[-2]
                            # record results
                            results.append(pred_type)

                        eae_instance_idx += 1

        assert len(preds) == eae_instance_idx
        
    pos_labels = list(data_args.role2id.keys())
    pos_labels.remove("NA")
    micro_f1 = f1_score(all_labels, results, labels=pos_labels, average="micro") * 100.0

    print('Evaluate Mode: {}, Number of Instances: {}'.format(eval_mode, eae_instance_idx))
    print("After converting, the micro_f1 is %.4f" % micro_f1)
    return results