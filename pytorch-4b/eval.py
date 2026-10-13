#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Evaluate a checkpoint written by train.py.

Reports:
  * mean next-token cross-entropy and perplexity over the FULL val split
    (contiguous, non-overlapping seq_len windows; same scheme as
    train.py's evaluate()),
  * a unigram baseline (train-split token frequencies, add-alpha smoothed
    over the full vocab) scored on exactly the same val target tokens,
  * greedy and temperature samples for each prompt in data/re-prompts.txt,
    decoded with tokenizer/tokenizer.json.

Usage:
    python3 eval.py --ckpt runs/small-cpu/best.pt [--json out.json]
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import Transformer4B, CONFIGS

HERE = pathlib.Path(__file__).resolve().parent


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CONFIGS["small"]()
    for k, v in ck["config"].items():
        setattr(cfg, k, v)
    model = Transformer4B(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    del ck["model"]
    ck.pop("optimizer", None)
    return model, cfg, ck


def window_targets(data, seq_len):
    n = (len(data) - 1) // seq_len
    return n, n * seq_len


@torch.no_grad()
def val_loss(model, data, seq_len, vocab, batch_size):
    n, _ = window_targets(data, seq_len)
    tot, cnt = 0.0, 0
    for b in range(0, n, batch_size):
        starts = [i * seq_len for i in range(b, min(n, b + batch_size))]
        x = torch.stack([torch.from_numpy(data[s:s + seq_len].astype("int64")) for s in starts])
        y = torch.stack([torch.from_numpy(data[s + 1:s + seq_len + 1].astype("int64")) for s in starts])
        loss = F.cross_entropy(model(x).view(-1, vocab), y.view(-1), reduction="sum")
        tot += loss.item()
        cnt += y.numel()
    return tot / cnt, cnt


def unigram_baseline(train, val, seq_len, vocab, alpha):
    counts = np.bincount(train, minlength=vocab).astype(np.float64)
    probs = (counts + alpha) / (counts.sum() + alpha * vocab)
    _, m = window_targets(val, seq_len)
    tgt = np.asarray(val[1:m + 1], dtype=np.int64)  # identical targets to val_loss
    nll = -np.log(probs[tgt]).mean()
    oov = float((counts[tgt] == 0).mean())
    return float(nll), int(tgt.size), oov


@torch.no_grad()
def generate(model, ids, max_new, temperature, ctx, eos_id, gen):
    ids = list(ids)
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-ctx:]], dtype=torch.long)
        logits = model(x)[0, -1]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            p = torch.softmax(logits / temperature, -1)
            nxt = int(torch.multinomial(p, 1, generator=gen))
        if nxt == eos_id:
            break
        ids.append(nxt)
        out.append(nxt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-data", default=str(HERE / "data/val.bin"))
    ap.add_argument("--train-data", default=str(HERE / "data/tokens.bin"))
    ap.add_argument("--prompts", default=str(HERE / "data/re-prompts.txt"))
    ap.add_argument("--tokenizer", default=str(HERE / "tokenizer/tokenizer.json"))
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--unigram-alpha", type=float, default=1.0)
    ap.add_argument("--eos-id", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    model, cfg, meta = load_model(args.ckpt)
    val = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    train = np.memmap(args.train_data, dtype=np.uint16, mode="r")

    t0 = time.time()
    vl, n_tok = val_loss(model, val, args.seq_len, cfg.vocab_size, args.batch_size)
    ul, u_tok, oov = unigram_baseline(train, val, args.seq_len, cfg.vocab_size, args.unigram_alpha)
    res = {
        "ckpt": str(args.ckpt), "step": meta.get("step"),
        "ckpt_sampled_val_loss": meta.get("val_loss"),
        "params": model.param_count(),
        "val_tokens_scored": n_tok, "seq_len": args.seq_len,
        "val_loss": round(vl, 4), "val_ppl": round(math.exp(vl), 2),
        "unigram_alpha": args.unigram_alpha,
        "unigram_val_loss": round(ul, 4), "unigram_val_ppl": round(math.exp(ul), 2),
        "val_targets_unseen_in_train": round(oov, 5),
        "eval_seconds": round(time.time() - t0, 1),
        "samples": [],
    }
    print(f"checkpoint {args.ckpt} (step {res['step']}, {res['params']:,} params)")
    print(f"val: {n_tok:,} tokens | loss {vl:.4f} | ppl {math.exp(vl):.2f}")
    print(f"unigram baseline (alpha={args.unigram_alpha}): loss {ul:.4f} | ppl {math.exp(ul):.2f}"
          f" | val targets unseen in train {oov:.4%}")

    prompts = [l.rstrip("\n") for l in open(args.prompts) if l.strip()]
    if args.n_prompts:
        prompts = prompts[:args.n_prompts]
    gen = torch.Generator().manual_seed(args.seed)
    for p in prompts:
        ids = tok.encode(p).ids
        g = tok.decode(generate(model, ids, args.max_new, 0.0, args.seq_len, args.eos_id, gen))
        s = tok.decode(generate(model, ids, args.max_new, args.temperature, args.seq_len, args.eos_id, gen))
        res["samples"].append({"prompt": p, "greedy": g, f"t{args.temperature}": s})
        print(f"\n### {p}\n[greedy] {g}\n[T={args.temperature}] {s}", flush=True)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
