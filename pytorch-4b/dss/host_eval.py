#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Score live llama.cpp runs made by the Rust host (dss-host eval).

Usage: host_eval.py MODE GGUF TEST.jsonl OUT_PREFIX
Runs `dss-host eval`, then grades each final answer with grade.py (an offline
scorer; it plays no part at inference time). Prints per-task and overall
well-formed call rate, correct-tool rate, copy-error rate and accuracy ± SE.
"""
import csv, json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grade import grade

mode, gguf, test, prefix = sys.argv[1:5]
rows = [json.loads(l) for l in open(HERE / test)]
pfile = pathlib.Path(prefix + ".prompts.txt"); tsv = pathlib.Path(prefix + ".tsv")
pfile.write_text("".join(r["prompt"] + "\n" for r in rows))
subprocess.run([str(HERE / "host/target/release/dss-host"), "eval", "--mode", mode, "--gguf", gguf,
                str(pfile), str(tsv)], check=True)
out = list(csv.DictReader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE))
assert len(out) == len(rows)
by = {}
for r, o in zip(rows, out):
    ok = o["wellformed"] == "1" and grade(r["prompt"], o["final"]) == 1.0
    by.setdefault(r["task"], []).append((o, ok))
    if r.get("src") == "live-failure":
        print("live-failure prompt:", o["text"], "| correct" if ok else "| WRONG")
res = {}
for name, items in list(by.items()) + [("all", [x for v in by.values() for x in v])]:
    n = len(items); p = sum(ok for _, ok in items) / n
    res[name] = {"n": n, "wellformed": round(sum(o["wellformed"] == "1" for o, _ in items) / n, 4),
                 "correct_tool": round(sum(o["correct_tool"] == "1" for o, _ in items) / n, 4),
                 "copy_error": round(sum(o["copy_error"] == "1" for o, _ in items) / n, 4),
                 "acc": round(p, 4), "acc_se": round((p * (1 - p) / n) ** 0.5, 4)}
json.dump(res, open(prefix + ".json", "w"), indent=1)
print(json.dumps({"mode": mode, "gguf": gguf, "test": test, **res}))
