#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Train a BPE tokenizer on the RE corpus.

Trains a 32k-vocab BPE tokenizer on pytorch-4b/data/full-corpus.jsonl.
Output is a HuggingFace tokenizers JSON file.

Usage:
    python3 scripts/train_tokenizer.py --out pytorch-4b/tokenizer/
"""
import argparse
import json
import pathlib

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def iter_texts(corpus_path, limit=None):
    count = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            if len(text.strip()) < 50:
                continue
            yield text
            count += 1
            if limit and count >= limit:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("pytorch-4b/data/full-corpus.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max documents (for testing)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
    )

    print(f"Training BPE ({args.vocab_size} vocab) on {args.corpus}...",
          flush=True)
    tok.train_from_iterator(iter_texts(args.corpus, args.limit), trainer)

    out_path = args.out / "tokenizer.json"
    tok.save(str(out_path))
    print(f"Saved: {out_path}")
    print(f"Vocab size: {tok.get_vocab_size()}")


if __name__ == "__main__":
    raise SystemExit(main())
