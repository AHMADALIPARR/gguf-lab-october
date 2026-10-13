#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Greedy/temperature sampling from a pytorch-4b checkpoint (PyTorch side).

Usage:
    python3 scripts/sample_small.py runs/small-cpu/best.pt "prompt" [--n 60]
"""
import argparse
import pathlib
import sys

import torch
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pytorch-4b"))
from model import Transformer4B, Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = Config()
    for k, v in ck["config"].items():
        setattr(cfg, k, v)
    m = Transformer4B(cfg)
    m.load_state_dict(ck["model"])
    m.eval()
    tok = Tokenizer.from_file(str(ROOT / "pytorch-4b/tokenizer/tokenizer.json"))
    ids = tok.encode(a.prompt).ids
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(a.n):
            x = torch.tensor([ids[-cfg.max_seq_len:]])
            logits = m(x)[0, -1]
            if a.temp <= 0:
                nxt = int(logits.argmax())
            else:
                nxt = int(torch.multinomial(torch.softmax(logits / a.temp, -1), 1))
            ids.append(nxt)
            if nxt == 2:
                break
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
