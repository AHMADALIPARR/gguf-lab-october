#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Greedy answer parity: llama-simple on a GGUF vs PyTorch on the checkpoint.

Usage: gguf_parity.py CKPT GGUF [--n 40]. Compares the first answer line for
the first N/4 prompts of each task in test_balanced.jsonl."""
import argparse, json, pathlib, subprocess, sys
import torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from eval import load_model
from tokenizers import Tokenizer
import loop


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("gguf")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--test", default="data/test_balanced.jsonl")
    ap.add_argument("--llama-bin", default="/workspace/llama.cpp/build/bin")
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok = Tokenizer.from_file(str(HERE.parent / "tokenizer/tokenizer.json"))
    loop.NL_IDS.update(tok.encode("\n").ids)
    model, _, _ = load_model(a.ckpt)
    rows = [json.loads(l) for l in open(HERE / a.test)]
    by = {}
    for r in rows: by.setdefault(r["task"], []).append(r)
    pick = [r for v in by.values() for r in v[: a.n // len(by)]]
    same = 0
    for r in pick:
        pt = tok.decode(loop.sample(model, tok.encode(r["prompt"]).ids, 1, 0.0, a.max_new)[0]).split("\n")[0]
        out = subprocess.run([f"{a.llama_bin}/llama-simple", "-m", a.gguf, "-n", str(a.max_new), r["prompt"]],
                             capture_output=True, text=True).stdout
        ll = out[len(r["prompt"]):].split("\n")[0] if out.startswith(r["prompt"]) else out.split("\n")[0]
        same += pt == ll
        if pt != ll: print("DIFF", repr(r["prompt"]), repr(pt), repr(ll))
    print(f"parity: {same}/{len(pick)} identical greedy answers")


if __name__ == "__main__":
    main()
