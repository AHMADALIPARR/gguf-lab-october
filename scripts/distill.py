#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Sequence-level distillation: generate with teacher, train 4B on output.

Sidesteps the vocab mismatch (teachers use 98k-262k vocab, 4B uses 32k)
by distilling at the text level, not the logit level.

Pipeline:
  1. Generate: feed prompts to teacher via llama.cpp/Ollama HTTP API
  2. Collect: teacher responses -> JSONL
  3. Tokenize: with 4B tokenizer -> tokens.bin
  4. Train: standard train.py on the distilled tokens

Usage:
    # Step 1: generate (needs teacher running, e.g. llama-server)
    python3 scripts/distill.py --generate \\
        --teacher-url http://localhost:8080 \\
        --prompts prompts/re-prompts.txt \\
        --out pytorch-4b/data/distilled.jsonl

    # Step 2: tokenize + train (see train.py)
"""
import argparse
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def query_teacher(url, prompt, max_tokens=512):
    """Query llama-server /completion endpoint."""
    data = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stop": ["</s>"],
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--teacher-url", default="http://localhost:8080")
    ap.add_argument("--prompts", type=pathlib.Path, required=True,
                    help="Text file, one prompt per line")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8")
               if l.strip()]
    print(f"{len(prompts)} prompts", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            try:
                resp = query_teacher(args.teacher_url, prompt, args.max_tokens)
            except Exception as e:
                print(f"[{i}] teacher error: {e}", flush=True)
                continue
            # Training format: prompt + response as one document
            text = f"{prompt}\n\n{resp}"
            f.write(json.dumps({"text": text, "prompt": prompt}) + "\n")
            n += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(prompts)} ({n} ok)", flush=True)

    print(f"Distilled {n} documents -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
