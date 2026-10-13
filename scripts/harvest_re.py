#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Harvest reverse-engineering training text from workspace repos.

Collects RE-relevant text (disassembly, binary analysis, unary semantics,
Alloy specs, pipeline docs) from rea-unary, hologram-re, and related
projects into a single training corpus.

Output: JSONL with {"text": ...} per document, ready for tokenization.

Usage:
    python3 scripts/harvest_re.py --out data/re-corpus.jsonl
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

# Repos and globs to harvest (relative to workspace)
SOURCES = [
    ("rea-unary", ["stage1/*.md", "stage1/*.txt", "stage1/*.jsonl",
                    "stage2/*.md", "stage3/*.md"]),
    ("hologram-re", ["functions/*.md", "alloy/*.als",
                      "oracle/README.md"]),
    ("hologram-parametric", ["README.md"]),
]

# Skip binary, vendored, or node_modules content
SKIP = ["node_modules", ".git/", "target/", "LICENSE"]


def should_skip(path):
    s = str(path)
    return any(skip in s for skip in SKIP)


def harvest():
    docs = []
    for repo, globs in SOURCES:
        repo_path = WORKSPACE / repo
        if not repo_path.exists():
            print(f"skip {repo}: not found", flush=True)
            continue
        for pattern in globs:
            for path in sorted(repo_path.glob(pattern)):
                if should_skip(path) or not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                # Skip tiny files
                if len(text.strip()) < 100:
                    continue
                docs.append({
                    "source": f"{repo}/{path.relative_to(repo_path)}",
                    "chars": len(text),
                    "text": text,
                })
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    docs = harvest()
    total_chars = sum(d["chars"] for d in docs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d["text"],
                                "source": d["source"]}) + "\n")

    print(f"Documents: {len(docs)}")
    print(f"Total chars: {total_chars:,}")
    print(f"Est. tokens (~4 chars/token): {total_chars // 4:,}")
    print(f"Output: {args.out}")

    # Honest assessment
    tokens = total_chars // 4
    needed = 4_000_000_000 * 20  # Chinchilla: ~20 tokens/param
    print(f"\nFor 4B model (Chinchilla): ~{needed:,} tokens needed")
    print(f"Have: {tokens:,} tokens ({100*tokens/needed:.4f}%)")
    if tokens < needed:
        print("INSUFFICIENT for from-scratch training. "
              "Fine-tune only, or gather more data.")


if __name__ == "__main__":
    raise SystemExit(main())
