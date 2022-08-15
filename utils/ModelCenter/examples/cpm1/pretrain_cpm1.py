import time
import random
import torch
import bmtrain as bmt
import numpy as np
import os

from model_center.model import CPM1Config, CPM1
from model_center.tokenizer import CPM1Tokenizer
from model_center.dataset import DistributedMMapIndexedDataset, MMapIndexedDataset
from model_center.dataset.cpm1dataset import CPM1_Dataset, CPM1_Dataset_Merge
from model_center import get_args
from model_center.utils import print_inspect

from torch.utils.tensorboard import SummaryWriter

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    config = CPM1Config.from_pretrained(args.model_config)
    model = CPM1(config)
    if args.load != None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
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
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
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
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def batch_iter(args, dataset, start_step = 0):
    st = 0
    ctx = []
    tgt = []
    context = []
    span = []
    hx = 0
    while True:
        ctx_data, tgt_data, _len, context_data = dataset[st]
        st += 1
        if ctx_data is None:
            continue
        assert _len <= args.max_length

        ctx_data = ctx_data.astype("int64")
        tgt_data = tgt_data.astype("int64")

        for index in range(len(ctx)):
            if span[index][-1] + _len < args.max_length:
                ctx[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(ctx_data)[:_len].long()
                tgt[index][span[index][-1]:span[index][-1] + _len]= torch.from_numpy(tgt_data)[:_len].long()
                context[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(context_data)[:_len].bool()
                span[index].append(span[index][-1] + _len)
                break
        else:
            _ctx = torch.zeros((args.max_length,), dtype=torch.long)
            _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
            _tgt = torch.full((args.max_length,), -100, dtype=torch.long)
            _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
            _context = torch.full((args.max_length,), False, dtype=torch.bool)
            _context[:_len] = torch.from_numpy(context_data)[:_len].bool()

            ctx.append(_ctx)
            tgt.append(_tgt)
            context.append(_context)
            span.append([_len])

        if len(ctx) > args.batch_size:
            if hx >= start_step:

                _span = torch.zeros((args.batch_size, args.max_length + 1), dtype=torch.long)
                for bindex in range(args.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1

                yield {
                    "ctx": torch.stack(ctx[:args.batch_size]),
                    "tgt": torch.stack(tgt[:args.batch_size]),
                    "context": torch.stack(context[:args.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:,:-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[:args.batch_size]]),
                }

            hx += 1
            ctx = ctx[args.batch_size:]
            tgt = tgt[args.batch_size:]
            context = context[args.batch_size:]
            span = span[args.batch_size:]


def pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    loss_func_tmp = torch.nn.CrossEntropyLoss(ignore_index=-100, reduce = False)


    if bmt.rank() == 0:
        writer = SummaryWriter("runs/cpm-1")

    start_step = args.start_step

    for iteration, data in enumerate(batch_iter(args, dataset, start_step)):
        iteration = iteration + start_step

        st = time.time()
        optimizer.zero_grad()

        assert len(data["ctx"]) == args.batch_size

        input_idx = data["ctx"].int().cuda()
        input_length = data["len_ctx"].int().cuda()
        input_context = data["context"].bool().cuda()
        input_span = data["span"].int().cuda()
        targets = data["tgt"].long().cuda()

        logits = model(input_idx, input_length, input_context, input_span)
        # loss_1 = loss_func_tmp(logits.view(-1, logits.size(-1)), targets.view(-1))
        # print (loss_1.max(), "==========", (loss_1>10).sum(), loss_1.mean())
        
        loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
        global_loss = bmt.sum_loss(loss).item()

        loss = optimizer.loss_scale(loss)
        loss.backward()
        grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)

        bmt.optim_step(optimizer, lr_scheduler)

        iteration_time = time.time() - st
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

        bmt.print_rank(
                "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} | token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}".format(
                    iteration,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    average_time / (1 - pow(average_time_shift, iteration + 1)),
                    input_length.float().mean()/args.max_length,
                    (targets>=0).sum(-1).float().mean()/args.max_length,
                    grad_norm
                )
            )

        if iteration % args.inspect_iters == 0:
            print_inspect(model, "*")
        if bmt.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration + start_step)
        if args.save != None and iteration % args.save_iters == 0:
            bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % iteration)))

    bmt.save(model, os.path.join(args.save, args.save_name+".pt"))

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = CPM1_Dataset_Merge(
        DistributedMMapIndexedDataset("/mnt/sfs_turbo/hx/ModelCenter/new_data/", "cpm1_lm_document_context", bmt.rank(), bmt.world_size()), 
        args.max_length
    )
    pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
