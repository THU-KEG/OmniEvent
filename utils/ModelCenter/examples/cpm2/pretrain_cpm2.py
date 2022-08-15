import time
import random
import torch
import bmtrain as bmt
import numpy as np
import os

from model_center import get_args
from model_center.model import CPM2Config, CPM2
from model_center.tokenizer import CPM2Tokenizer
from model_center.dataset import DistributedMMapIndexedDataset, MMapIndexedDataset
from model_center.dataset.cpm2dataset import CPM2_Dataset
from model_center.utils import print_inspect

import distutils.version
from torch.utils.tensorboard import SummaryWriter

def get_tokenizer(args):
    tokenizer = CPM2Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    config = CPM2Config.from_pretrained(args.model_config)
    model = CPM2(config)
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


def pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    if bmt.rank() == 0:
        writer = SummaryWriter("runs/cpm-2")
    
    start_step = args.start_step

    for iteration, data in enumerate(dataset):
        iteration = iteration + start_step

        st = time.time()
        optimizer.zero_grad()

        assert len(data["ctx"]) == args.batch_size

        enc_input = data["ctx"].int().cuda()
        enc_length = data["len_ctx"].int().cuda()
        dec_input = torch.clamp(data["tgt"].int()[:, :-1], 0).cuda()
        targets = data["tgt"].long()[:, 1:].cuda()
        dec_length = data["len_tgt"].int().cuda()

        logits = model(enc_input, enc_length, dec_input, dec_length)
        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
        global_loss = bmt.sum_loss(loss).item()

        loss = optimizer.loss_scale(loss)
        loss.backward()

        bmt.optim_step(optimizer, lr_scheduler)

        iteration_time = time.time() - st
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                global_loss,
                lr_scheduler.current_lr,
                int(optimizer.scale),
                average_time / (1 - pow(average_time_shift, iteration + 1)),
            )
        )
        if iteration % args.inspect_iters == 0:
            print_inspect(model, "*")
        if bmt.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration + start_step)
        if args.save != None and iteration % args.save_iters == 0:
            bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % iteration)))

    bmt.save(model, os.path.join(args.save, args.save_name+".pt"))

class ShuffleDataset(torch.utils.data.IterableDataset): # TODO
    def __init__(self, dataset : CPM2_Dataset, rank, world_size, shuffle_idx, st = 0):
        self.rank = rank
        self.world_size = world_size
        self.shuffle_idx = shuffle_idx
        self.dataset = dataset
        self.st = st
        self.end = len(shuffle_idx)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError("Multi-worker not supported")
        while self.st < self.end:
            it = self.dataset[ int(self.shuffle_idx[ self.st + self.rank ]) ]
            if it is not None:
                yield it
            self.st += self.world_size

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = CPM2_Dataset(
        MMapIndexedDataset("/mnt/sfs_turbo/data0814/large_data/shuf_2_26_new_document_context"),
        MMapIndexedDataset("/mnt/sfs_turbo/data0814/large_data/shuf_2_26_new_document_target"),
        max_source_length = args.max_encoder_length,
        max_target_length = args.max_decoder_length,
    )

    shuf_idx = np.load("/mnt/sfs_turbo/zgy/cpm2_pretrain_new/shuffle_idx.npy")
    shuf_data = ShuffleDataset(dataset, bmt.rank(), bmt.world_size(), shuf_idx)

    dataloader = torch.utils.data.DataLoader(
        shuf_data, 
        batch_size = args.batch_size, 
        num_workers = 1
    )

    pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataloader)

if __name__ == "__main__":
    main()
