import time
import random
import os
import csv
import sys 
sys.path.append("../../")

import copy 
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import bmtrain as bmt

from model_center import get_args
from model_center.model import T5
from model_center.tokenizer import T5Tokenizer
from model_center.dataset.t5dataset import DATASET
from model_center.utils import print_inspect
from model_center.dataset import DistributedDataLoader

from OmniEvent.input_engineering.seq2seq_processor import EDSeq2SeqProcessor, extract_argument
from OmniEvent.evaluation.metric import f1_score_overall

from transformers import (
    LogitsProcessorList, 
    ForcedEOSTokenLogitsProcessor,
    BeamSearchScorer, 
    StoppingCriteriaList, 
    MaxLengthCriteria
)
from beam_search import beam_search


def get_tokenizer(model_config):
    tokenizer = T5Tokenizer.from_pretrained(model_config)
    return tokenizer

def get_model(model_config):
    model = T5.from_pretrained(model_config)
    return model

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args.model_config)
    # get the model
    model = get_model(args.model_config)
    # model.load_state_dict(torch.load(os.path.join(args.save, args.save_name)))
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 100)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer):
    dataset = {}
    dataset["train"] = EDSeq2SeqProcessor(args, tokenizer, args.train_file)
    dataset["dev"] = EDSeq2SeqProcessor(args, tokenizer, args.validation_file)
    dataset["test"] = EDSeq2SeqProcessor(args, tokenizer, args.test_file)
    return dataset


def compute_seq_F1(logits, labels, task_name, **kwargs):
    tokenizer = kwargs["tokenizer"]
    training_args = kwargs["training_args"]
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=False)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    if task_name == "EAE":
        pred_arguments, golden_arguments = [], []
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            pred = clean_str(pred)
            label = clean_str(label)
            pred_arguments.extend(extract_argument(pred, i, "NA"))
            golden_arguments.extend(extract_argument(label, i, "NA"))
        # import pdb; pdb.set_trace()
        precision, recall, micro_f1 = f1_score_overall(pred_arguments, golden_arguments)
    else:
        assert len(decoded_preds) == len(decoded_labels)
        pred_triggers, golden_triggers = [], []
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            pred = clean_str(pred)
            label = clean_str(label)
            pred_triggers.extend(extract_argument(pred, i, "NA"))
            golden_triggers.extend(extract_argument(label, i, "NA"))
        precision, recall, micro_f1 = f1_score_overall(pred_triggers, golden_triggers)
    return {"micro_f1": micro_f1*100}


def evaluate(args, model, tokenizer, dataloader, epoch, split):
    model.eval()
    with torch.no_grad():
        synced_gpus = True if torch.cuda.device_count() > 1 else False 
        bmt.print_rank("Synced_gpus: %s" % str(synced_gpus))
        num_beams = 4
        model.config.is_encoder_decoder = True 
        pd = []
        gt = []
        for it, data in enumerate(dataloader[split]):
            enc_input = data["input_ids"].cuda()
            target = copy.deepcopy(data["labels"]).cuda()
            # prepare for beam search 
            batch_size = enc_input.size(0)
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=args.local_rank,
            )
            logits_processor = LogitsProcessorList(
                [
                    ForcedEOSTokenLogitsProcessor(args.max_out_length, eos_token_id=1),
                ]
            )
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=args.max_out_length)])
            # input sequence 
            model_kwargs = {
                "encoder_outputs": model.encode(
                    enc_input.repeat_interleave(num_beams, dim=0),
                    data["attention_mask"].cuda().repeat_interleave(num_beams, dim=0)
                )
            }
            input_ids = torch.zeros((batch_size*num_beams, 1), device=args.local_rank, dtype=torch.long)
            outputs = beam_search(args=args,
                        config = model.config,
                        model = model, 
                        input_ids = input_ids, 
                        encoder_attention_mask = data["attention_mask"].cuda().repeat_interleave(num_beams, dim=0).to(torch.bool),
                        beam_scorer = beam_scorer, 
                        logits_processor = logits_processor,
                        stopping_criteria = stopping_criteria,
                        max_length = args.max_out_length,
                        synced_gpus = synced_gpus,
                        **model_kwargs)
            for sequence in outputs.cpu().tolist(): 
                while len(sequence) < args.max_out_length:
                    sequence.append(0)
                pd.append(sequence)
            gt.extend(target.cpu().tolist())

            bmt.print_rank(
                "{} | epoch {:3d} | Iter: {:6d}/{:6d} |".format(
                    split,
                    epoch,
                    it,
                    len(dataloader[split]),
                )
            )
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()
        
        bmt.print_rank(f"{split} epoch {epoch}:")
        mirco_f1 = compute_seq_F1(np.array(pd), np.array(gt), "EAE", **{"tokenizer": tokenizer, "training_args": args})
        bmt.print_rank(mirco_f1)

        return mirco_f1 



def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    dataloader_num_workers = 2
    best_f1 = 0
    start_dev_epoch = 1
    dev_epoch_step = 1
    for epoch in range(args.epochs):
        dataloader = {
            "train": DistributedDataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, **{"num_workers": dataloader_num_workers}),
            "dev": DistributedDataLoader(dataset['dev'], batch_size=32, shuffle=False, **{"num_workers": dataloader_num_workers}),
            "test": DistributedDataLoader(dataset['test'], batch_size=32, shuffle=False, **{"num_workers": dataloader_num_workers})
        }

        if args.do_train:
            model.train()
            for it, data in enumerate(dataloader['train']):
                enc_input = data["input_ids"].cuda()
                enc_length = data["attention_mask"].cuda().sum(-1).to(torch.int32)
                target = copy.deepcopy(data["labels"]).cuda()
                dec_input = data["labels"].cuda()
                dec_length = (data["labels"]!=-100).cuda().sum(-1).to(torch.int32)
                dec_input = (dec_input!=-100) * dec_input

                optimizer.zero_grad()

                # import pdb; pdb.set_trace()
                logits = model(enc_input, enc_length, dec_input, dec_length, output_logits=True).logits

                logits = logits.reshape(-1, logits.size(-1))
                target = target.reshape(-1)
                loss = loss_func(logits, target)
                global_loss = bmt.sum_loss(loss).item()

                loss = optimizer.loss_scale(loss)
                loss.backward()
                grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)

                bmt.optim_step(optimizer, lr_scheduler)

                bmt.print_rank(
                    "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                        epoch,
                        it,
                        len(dataloader["train"]),
                        global_loss,
                        lr_scheduler.current_lr,
                        int(optimizer.scale),
                        grad_norm,
                    )
                )

        if epoch < start_dev_epoch or epoch % dev_epoch_step != 0:
            continue
        micro_f1 = evaluate(args, model, tokenizer, dataloader, epoch, "dev")
        if micro_f1["micro_f1"] > best_f1:
            best_f1 = micro_f1["micro_f1"]
            bmt.print_rank("Best dev score. Saving...")
            # save 
            bmt.save(model, os.path.join(args.save, args.save_name))
            if args.local_rank == 0:
                os.system(f"cp {args.model_config}/*.json {args.save}")
    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(args.save, args.save_name)))
        model = get_model(args.save)
        bmt.synchronize()
        bmt.print_rank("Best Dev F1: %.2f, Testing..." % best_f1)
        evaluate(args, model, tokenizer, dataloader, epoch, "test")



def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
