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

from model import Transformer4B, CONFIGS, config_to_dict


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


@torch.no_grad()
def evaluate(model, data, seq_len, batch_size, vocab, max_batches):
    """Mean next-token loss over contiguous, non-overlapping val windows."""
    model.eval()
    n = (len(data) - 1) // seq_len
    starts = list(range(0, n * seq_len, seq_len))[:max_batches * batch_size]
    tot, cnt = 0.0, 0
    for b in range(0, len(starts), batch_size):
        sb = starts[b:b + batch_size]
        x = torch.stack([torch.from_numpy(data[s:s + seq_len].astype("int64")) for s in sb])
        y = torch.stack([torch.from_numpy(data[s + 1:s + seq_len + 1].astype("int64")) for s in sb])
        loss = nn.functional.cross_entropy(model(x).view(-1, vocab), y.view(-1))
        tot += loss.item() * len(sb)
        cnt += len(sb)
    model.train()
    return tot / max(1, cnt)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU — training on CPU will be extremely slow")

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = CONFIGS[args.config]()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    model = Transformer4B(cfg).to(device)
    n_params = model.param_count()
    print(f"Model: {n_params:,} params ({n_params/1e9:.2f}B)")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay)

    ds = TokenDataset(args.data, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, drop_last=True)
    val = None
    if args.val_data:
        import numpy as np
        val = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    print(f"Train sequences: {len(ds)} x {args.seq_len} tokens", flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.csv", "a")
    log.write("step,epoch,train_loss,val_loss,lr,elapsed_s\n")
    best_val = float("inf")
    t_start = time.time()

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
                vl = ""
                if val is not None and step % args.eval_every == 0:
                    v = evaluate(model, val, args.seq_len, args.batch_size,
                                 cfg.vocab_size, args.eval_batches)
                    vl = f"{v:.4f}"
                    if v < best_val:
                        best_val = v
                        torch.save({"step": step, "val_loss": v,
                                    "model": model.state_dict(),
                                    "config": config_to_dict(cfg)},
                                   out / "best.pt")
                el = time.time() - t_start
                print(f"step {step:>7} | ep {epoch} | loss {avg:.4f} | "
                      f"val {vl or '-'} | lr {lr:.2e} | {dt:.0f}s | "
                      f"total {el:.0f}s", flush=True)
                log.write(f"{step},{epoch},{avg:.4f},{vl},{lr:.3e},{el:.1f}\n")
                log.flush()
                total_loss = 0.0
                t0 = time.time()

            if step % args.ckpt_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": config_to_dict(cfg),
                }
                path = out / f"ckpt-{step:07d}.pt"
                torch.save(ckpt, path)
                print(f"checkpoint: {path}", flush=True)

            if step >= args.total_steps:
                break
        if step >= args.total_steps:
            break

    # Final save
    final_val = None
    if val is not None:
        final_val = evaluate(model, val, args.seq_len, args.batch_size,
                             cfg.vocab_size, 10**9)
        print(f"final full val loss: {final_val:.4f}", flush=True)
    torch.save({"step": step, "model": model.state_dict(),
                "config": config_to_dict(cfg)}, out / "final.pt")
    # Training report
    (out / "report.json").write_text(json.dumps({
        "params": n_params,
        "steps": step,
        "epochs_started": epoch + 1,
        "wall_time_s": round(time.time() - t_start, 1),
        "final_val_loss_full": final_val,
        "best_val_loss_sampled": best_val if val is not None else None,
        "args": vars(args),
        "config": config_to_dict(cfg),
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
    ap.add_argument("--config", choices=sorted(CONFIGS), default="4b")
    ap.add_argument("--val-data", default=None, help="Held-out .bin (uint16)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())
