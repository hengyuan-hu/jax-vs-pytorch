import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable

from config import ModelConfig
from data import Enwik9Loader
from pytorch_model import *


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


def train_epoch(lm, cfg: ModelConfig, datapath: str) -> None:
    optimizer = torch.optim.Adam(lm.parameters(), cfg.learning_rate)
    losses = []
    t = time.time()
    for i, batch in enumerate(Enwik9Loader(cfg.batch_size, cfg.seq_len, datapath)):
        loss = compute_loss(lm, torch.tensor(batch, device="cuda"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        log_per = 20
        if i % log_per == 0:
            time_elps = time.time() - t
            speed = log_per * cfg.batch_size / time_elps
            print(f"At iter {i}, loss {np.mean(losses):.4f}, Speed: {speed:.2f}")
            t = time.time()
            losses = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, help="hand_craft/torch")
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--fp32_precision", type=str, default="highest")
    args = parser.parse_args()

    enwik9 = "./enwik9"

    # This is the highest batch size PyTorch can handle, the JAX model can do 79
    cfg = ModelConfig(
        seq_len=256,
        n_layers=1,
        d_model=512,
        num_heads=8,
        ff_dim=3072,
        dropout=0.1,
        batch_size=100,
        learning_rate=1e-3,
    )

    if args.model == "hand_craft":
        lm = HandCraftLM(cfg)
    else:
        lm = LM(cfg)
    lm = lm.cuda()
    print(lm)
    count_parameters(lm)

    if args.compile:
        lm = torch.compile(lm)

    for v in lm.parameters():
        print(
            f"{v.size()}".ljust(30),
            f"{abs(v.mean().item()):.2e}",
            f"{v.std().item():.2e}",
        )

    train_epoch(lm, cfg, enwik9)
