#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the v2 corpus: re-corpus.jsonl plus text files from a
devflow-finance-twin checkout, then tokenize into train/val streams.

Extraction rules for the checkout:
  * skip .git, binaries (NUL byte in the first 8 KB or not valid UTF-8),
    lockfiles, files over 1 MB, and empty files;
  * skip REUSE-style `*.license` sidecar files (per-file SPDX boilerplate);
  * exact SHA-256 dedupe on the text, within the checkout and against
    re-corpus.jsonl (this collapses repeated LICENSE copies to one).

Split:
  * val.bin     - the same 32 re-corpus docs as v1 (seeded shuffle, seed 0,
                  first 5% of the 650 docs), so v1 and v2 are comparable;
  * val-new.bin - 5% of the new devflow docs (seeded shuffle, seed 1);
  * tokens.bin  - everything else.
EOS (id 2) is appended after each doc. Stats go to data/split_stats.json.

Usage:
    python3 scripts/build_corpus_v2.py --src /workspace/devflow-finance-twin
"""
import argparse
import hashlib
import json
import os
import pathlib
import random

import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "pytorch-4b" / "data"
LOCKFILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock",
             "poetry.lock", "Pipfile.lock", "go.sum", "composer.lock",
             "Gemfile.lock", "flake.lock", "uv.lock", "bun.lockb"}


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def extract(src, seen, skips):
    docs = []
    for dp, dns, fns in os.walk(src):
        dns[:] = sorted(d for d in dns if d != ".git")
        for fn in sorted(fns):
            p = pathlib.Path(dp) / fn
            rel = str(p.relative_to(src))
            if fn in LOCKFILES or fn.endswith(".lock"):
                skips["lockfile"] += 1
                continue
            if fn.endswith(".license"):
                skips["license_sidecar"] += 1
                continue
            if p.is_symlink() or not p.is_file():
                skips["not_regular"] += 1
                continue
            if p.stat().st_size > 1_000_000:
                skips["over_1mb"] += 1
                continue
            raw = p.read_bytes()
            if b"\x00" in raw[:8192]:
                skips["binary"] += 1
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                skips["binary"] += 1
                continue
            if not text.strip():
                skips["empty"] += 1
                continue
            h = sha(text)
            if h in seen:
                skips["duplicate"] += 1
                continue
            seen.add(h)
            docs.append({"text": text, "source": f"devflow-finance-twin/{rel}"})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()
    src = pathlib.Path(a.src)

    old = []
    for line in open(D / "re-corpus.jsonl", encoding="utf-8"):
        try:
            old.append(json.loads(line)["text"])
        except Exception:
            continue
    seen = {sha(t) for t in old}
    skips = {k: 0 for k in ("lockfile", "license_sidecar", "not_regular",
                            "over_1mb", "binary", "empty", "duplicate")}
    new = extract(src, seen, skips)
    with open(D / "devflow-corpus.jsonl", "w", encoding="utf-8") as f:
        for d in new:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    idx = list(range(len(old)))
    random.Random(0).shuffle(idx)  # identical to v1 tokenize_split.py
    old_val = set(idx[:max(1, round(len(old) * a.val_frac))])
    nidx = list(range(len(new)))
    random.Random(1).shuffle(nidx)
    new_val = set(nidx[:max(1, round(len(new) * a.val_frac))])

    tok = Tokenizer.from_file(str(ROOT / "pytorch-4b/tokenizer/tokenizer.json"))
    eos = tok.token_to_id("</s>")
    outs = {k: open(D / f, "wb") for k, f in
            (("train", "tokens.bin"), ("val", "val.bin"), ("val_new", "val-new.bin"))}
    toks = {k: 0 for k in outs}
    ndocs = {k: 0 for k in outs}
    by_src = {"re_train": 0, "devflow_train": 0}

    def put(split, text, srckey=None):
        ids = [t for t in tok.encode(text).ids if t < 65536] + [eos]
        np.array(ids, dtype=np.uint16).tofile(outs[split])
        toks[split] += len(ids)
        ndocs[split] += 1
        if srckey:
            by_src[srckey] += len(ids)

    for i, t in enumerate(old):
        if i in old_val:
            put("val", t)
        else:
            put("train", t, "re_train")
    for i, d in enumerate(new):
        if i in new_val:
            put("val_new", d["text"])
        else:
            put("train", d["text"], "devflow_train")
    for f in outs.values():
        f.close()
    stats = {"vocab": tok.get_vocab_size(), "eos_id": eos,
             "re_corpus_docs": len(old), "devflow_new_docs": len(new),
             "devflow_skipped": skips, "docs": ndocs, "tokens": toks,
             "train_tokens_by_source": by_src}
    (D / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
