import argparse
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable

from config import ModelConfig
from data import Enwik9Loader
from pytorch_model import *
from logger import Logger


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def compute_loss(lm, batch):
    probs = lm(batch)
    probs = probs.reshape(-1, 256)
    targets = batch.reshape(-1).long()
    return nn.CrossEntropyLoss()(probs, targets)


def train_epoch(lm, cfg: ModelConfig, datapath: str, pt_dtype) -> None:
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.Adam(lm.parameters(), cfg.learning_rate)

    losses = []
    t = time.time()
    dataloader = list(Enwik9Loader(cfg.batch_size, cfg.seq_len, datapath))
    for i, batch in enumerate(dataloader):
        data = torch.tensor(batch, device="cuda").transpose(0, 1).contiguous()
        with torch.amp.autocast(device_type="cuda", dtype=pt_dtype):
            loss = compute_loss(lm, data)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        losses.append(loss.item())
        log_per = 20
        if (i + 1) % log_per == 0:
            torch.cuda.synchronize()
            time_elps = time.time() - t
            speed = log_per * cfg.batch_size / time_elps
            print(f"At iter {i+1}/{len(dataloader)}, loss: {np.mean(losses):.4f}, Speed: {speed:.2f}")
            t = time.time()
            losses = []

        if (i + 1) > cfg.max_num_batch:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--num_layer", type=int, default=24)
    parser.add_argument("--model", type=str, help="handcraft/torch")
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="exps/pytorch")
    # bfloat16 has the same range as float32, but different precision
    # float16 has less range as float32, but the same precision
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32/bfloat16/float16")
    parser.add_argument("--flash", type=int, default=0, help="flash attn")
    parser.add_argument("--xformer", type=int, default=0, help="xformer")
    args = parser.parse_args()

    # torch.set_float32_matmul_precision(args.fp32_precision)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    args.save_dir = f"{args.save_dir}_{args.model}_layer{args.num_layer}_{args.dtype}"
    if args.compile:
        args.save_dir = f"{args.save_dir}_compiled"
    if args.flash:
        args.save_dir = f"{args.save_dir}_flash"

    pt_dtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[args.dtype]

    logger_path = os.path.join(args.save_dir, f"train.log")
    sys.stdout = Logger(logger_path, print_to_stdout=True)
    logger_path = os.path.join(args.save_dir, f"train.log")
    print(f"writing to {logger_path}")
    sys.stdout = Logger(logger_path, print_to_stdout=True)

    enwik9 = "./enwik9"
    cfg = ModelConfig(
        seq_len=256,
        n_layers=args.num_layer,
        d_model=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        batch_size=100,
        learning_rate=1e-3,
        max_num_batch=5000,
    )

    if args.model == "handcraft":
        lm = HandCraftLM(cfg, args.flash, args.xformer)
    # elif args.model == "torch":
    #     lm = TorchLM(cfg)
    else:
        assert False

    lm = lm.cuda()
    print(lm)
    count_parameters(lm)

    if args.compile:
        lm = torch.compile(lm)

    # for v in lm.parameters():
    #     print(
    #         f"\t{v.size()}".ljust(30),
    #         f"{abs(v.mean().item()):.2e}",
    #         f"{v.std().item():.2e}",
    #     )

    train_epoch(lm, cfg, enwik9, pt_dtype)
