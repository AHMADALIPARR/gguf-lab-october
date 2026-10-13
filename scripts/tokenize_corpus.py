#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Tokenize the RE corpus into a flat binary file for training.

Reads pytorch-4b/data/full-corpus.jsonl, tokenizes with the trained
tokenizer, writes uint16 token ids to tokens.bin (matches TokenDataset
in train.py).

Usage:
    python3 scripts/tokenize_corpus.py
"""
import json
import pathlib

import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "pytorch-4b" / "data" / "full-corpus.jsonl"
TOK_PATH = ROOT / "pytorch-4b" / "tokenizer" / "tokenizer.json"
OUT = ROOT / "pytorch-4b" / "data" / "tokens.bin"


def main():
    tok = Tokenizer.from_file(str(TOK_PATH))
    print(f"Tokenizer vocab: {tok.get_vocab_size()}")

    # Stream tokenize to avoid holding everything in RAM
    total = 0
    with open(OUT, "wb") as out:
        with open(CORPUS, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    text = json.loads(line)["text"]
                except Exception:
                    continue
                ids = tok.encode(text).ids
                # Filter to uint16 range
                ids = [t for t in ids if t < 65536]
                if not ids:
                    continue
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(out)
                total += len(ids)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1} docs, {total:,} tokens", flush=True)

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"\nDone: {total:,} tokens -> {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    raise SystemExit(main())
