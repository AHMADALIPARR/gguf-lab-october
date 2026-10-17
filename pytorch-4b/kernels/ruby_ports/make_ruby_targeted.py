#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Targeted extra SFT rows for the two held-out zeros (viterbi, worm_seal).
Reuses the prompt generators of make_ruby_dataset.py (exec'd up to its main
loop, so nothing there is rewritten); answers come from `dss-host answer`.
Excludes hash-test prompts and every prompt already in ruby_{train,test,test_ood}.
Harness only. Writes dss/data/ruby_train_targeted.jsonl.
"""
import json, pathlib, random, sys
HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "make_ruby_dataset.py").read_text()
g = {"__file__": str(HERE / "make_ruby_dataset.py")}
exec(src.split("\ntrain, test, ood, seen = ")[0], g)
DATA = g["DATA"]; N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
seen = set()
for f in ("ruby_train.jsonl", "ruby_test.jsonl", "ruby_test_ood.jsonl"):
    seen |= {json.loads(l)["prompt"] for l in open(DATA / f)}
rng = random.Random(20261016); rows = []
for task in ("viterbi", "worm_seal"):
    n = tries = 0
    while n < N and tries < 400000:
        tries += 1
        p = g["gen_task"](task, rng, ood=False)
        if not g["fits"](p) or g["split"](p) == "test" or p in seen: continue
        seen.add(p); rows.append({"task": task, "prompt": p}); n += 1
    print(task, n, "tries", tries, flush=True)
for i in range(0, len(rows), 200):
    for r, a in zip(rows[i:i+200], g["answer"]([r["prompt"] for r in rows[i:i+200]])):
        r["answer"] = " " + a
rng.shuffle(rows)
with open(DATA / "ruby_train_targeted.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows))
