#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Tokenize pytorch-4b/data/re-corpus.jsonl into train/val uint16 streams.

Split is by document (deterministic shuffle, seed 0): ~5% of documents go
to val.bin, the rest to tokens.bin. An EOS id is appended after each doc
when the tokenizer defines one. Counts are written to split_stats.json.

Usage:
    python3 scripts/tokenize_split.py [--val-frac 0.05]
"""
import argparse
import json
import pathlib
import random

import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "pytorch-4b" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(D / "re-corpus.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(ROOT / "pytorch-4b/tokenizer/tokenizer.json"))
    eos = None
    for name in ("</s>", "<|endoftext|>", "<eos>"):
        if tok.token_to_id(name) is not None:
            eos = tok.token_to_id(name)
            break
    docs = []
    for line in open(a.corpus, encoding="utf-8"):
        try:
            docs.append(json.loads(line)["text"])
        except Exception:
            continue
    idx = list(range(len(docs)))
    random.Random(0).shuffle(idx)
    n_val = max(1, round(len(docs) * a.val_frac))
    val_set = set(idx[:n_val])
    counts = {"train": 0, "val": 0}
    ndocs = {"train": 0, "val": 0}
    with open(D / "tokens.bin", "wb") as ft, open(D / "val.bin", "wb") as fv:
        for i, text in enumerate(docs):
            ids = [t for t in tok.encode(text).ids if t < 65536]
            if eos is not None:
                ids.append(eos)
            split = "val" if i in val_set else "train"
            np.array(ids, dtype=np.uint16).tofile(fv if split == "val" else ft)
            counts[split] += len(ids)
            ndocs[split] += 1
    stats = {"vocab": tok.get_vocab_size(), "eos_id": eos,
             "docs": ndocs, "tokens": counts}
    (D / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
