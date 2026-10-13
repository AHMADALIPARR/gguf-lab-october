#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""PyTorch training loop for the 4B model.

Standard causal LM training: cross-entropy on next-token prediction,
AdamW, cosine schedule with warmup, gradient clipping, checkpointing.

Requires: GPU with >=40GB VRAM (or multi-GPU). This VM cannot run it.
Dataset interface is pluggable — provide your own tokenized data.

Usage:
    python3 train.py --data /path/to/tokens.bin --out checkpoints/
"""
import argparse
import json
import math
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import Transformer4B, Config


class TokenDataset(Dataset):
    """Memory-mapped token stream cut into fixed-length sequences.

    Expects a flat binary file of uint16 token ids.
    Replace with your own dataset class as needed.
    """

    def __init__(self, path, seq_len):
        import numpy as np
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        x = torch.from_numpy(
            self.data[s:s + self.seq_len].astype("int64"))
        y = torch.from_numpy(
            self.data[s + 1:s + self.seq_len + 1].astype("int64"))
        return x, y


def cosine_schedule(step, warmup, total, peak_lr, min_lr):
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (
        1 + math.cos(math.pi * progress))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU — training on CPU will be extremely slow")

    cfg = Config()
    model = Transformer4B(cfg).to(device)
    n_params = model.param_count()
    print(f"Model: {n_params:,} params ({n_params/1e9:.2f}B)")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay)

    ds = TokenDataset(args.data, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    total_loss = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            lr = cosine_schedule(
                step, args.warmup_steps, args.total_steps,
                args.peak_lr, args.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                avg = total_loss / args.log_every
                dt = time.time() - t0
                print(f"step {step:>7} | loss {avg:.4f} | "
                      f"lr {lr:.2e} | {dt:.0f}s", flush=True)
                total_loss = 0.0
                t0 = time.time()

            if step % args.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": {k: v for k, v in vars(cfg).items()
                               if not k.startswith("_")},
                }
                path = out / f"ckpt-{step:07d}.pt"
                torch.save(ckpt, path)
                print(f"checkpoint: {path}", flush=True)

            if step >= args.total_steps:
                break
        if step >= args.total_steps:
            break

    # Final save
    torch.save(model.state_dict(), out / "final.pt")
    # Training report
    (out / "report.json").write_text(json.dumps({
        "params": n_params,
        "steps": step,
        "config": {k: v for k, v in vars(cfg).items()
                   if not k.startswith("_")},
    }, indent=2))
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Token .bin file (uint16)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=100000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    train(ap.parse_args())
