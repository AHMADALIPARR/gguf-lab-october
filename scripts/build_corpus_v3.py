#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the v3 corpus: the v2 train stream (pytorch-4b/data/tokens.bin,
copied byte-for-byte, so every v2 train doc is kept) plus TinyStories
(roneneldan/TinyStories, Hugging Face) English prose, roughly 50/50 by tokens.

TinyStories source files (download via plain HTTPS into data/v3/raw/):
  train-00000.parquet  <- data/train-00000-of-00004-2d5a1467fff1081b.parquet
  validation.parquet   <- data/validation-00000-of-00001-869c898b519ad725.parquet

Train stories: shard 0 in file order, stripped, empty/duplicate (SHA-256)
dropped, taken until --story-tokens is reached. val-stories.bin: the first
--val-stories stories of the official validation split (seeded shuffle,
seed 2), excluding any exact duplicate of a train story.
EOS (id 2) after each story. v2 val.bin / val-new.bin are not touched.
Outputs in pytorch-4b/data/v3/: tokens.bin, val-stories.bin, split_stats.json.
"""
import argparse, hashlib, json, pathlib, random, shutil
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "pytorch-4b" / "data"
V3 = D / "v3"


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story-tokens", type=int, default=5_500_000)
    ap.add_argument("--val-stories", type=int, default=1000)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(ROOT / "pytorch-4b/tokenizer/tokenizer.json"))
    eos = tok.token_to_id("</s>")
    out = V3 / "tokens.bin"
    shutil.copyfile(D / "tokens.bin", out)
    v2_tokens = (D / "tokens.bin").stat().st_size // 2

    def enc(texts):
        return [[t for t in e.ids if t < 65536] + [eos]
                for e in tok.encode_batch(texts)]

    seen, n_story, st_tok, skipped = set(), 0, 0, 0
    texts = pq.read_table(V3 / "raw/train-00000.parquet", columns=["text"]).column("text").to_pylist()
    with open(out, "ab") as f:
        for i in range(0, len(texts), 2000):
            batch = []
            for t in texts[i:i + 2000]:
                t = (t or "").strip()
                h = sha(t) if t else None
                if not t or h in seen:
                    skipped += 1
                    continue
                seen.add(h)
                batch.append(t)
            for ids in enc(batch):
                np.array(ids, dtype=np.uint16).tofile(f)
                st_tok += len(ids); n_story += 1
                if st_tok >= a.story_tokens:
                    break
            if st_tok >= a.story_tokens:
                break
    val = [t.strip() for t in pq.read_table(V3 / "raw/validation.parquet", columns=["text"]).column("text").to_pylist() if t and t.strip()]
    random.Random(2).shuffle(val)
    val = [t for t in val if sha(t) not in seen][:a.val_stories]
    vt = 0
    with open(V3 / "val-stories.bin", "wb") as f:
        for ids in enc(val):
            np.array(ids, dtype=np.uint16).tofile(f); vt += len(ids)
    stats = {"vocab": tok.get_vocab_size(), "eos_id": eos,
             "train_tokens_by_source": {"v2_tokens_bin": v2_tokens, "tinystories": st_tok},
             "train_tokens_total": v2_tokens + st_tok,
             "tinystories_train_docs": n_story, "tinystories_skipped": skipped,
             "val_stories": {"docs": len(val), "tokens": vt}}
    (V3 / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
