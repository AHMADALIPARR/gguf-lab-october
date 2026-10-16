#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Summarise live llama.cpp runs (TSV files from dss-host eval) into one table.

Usage: host_summary.py RUN_DIR TEST_JSONL TSV [TSV ...]
For each TSV (restricted to prompts present in TEST_JSONL), prints n,
well-formed call rate, call copy-error rate, final-transcription error rate
(well-formed, but the final answer is not the tool result), and accuracy ± SE.
"""
import csv, json, sys
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
from grade import grade

test = {json.loads(l)["prompt"]: json.loads(l) for l in open(sys.argv[2])}
import grade as _G
for _r in test.values():
    if _r["task"].startswith("phi_") or _r["task"] in ("viterbi", "permission", "governance", "worm_seal"): _G.PHI_ANS[_r["prompt"]] = _r["answer"].strip()
for path in sys.argv[3:]:
    rows = [o for o in csv.DictReader(open(path), delimiter="\t", quoting=csv.QUOTE_NONE) if o["prompt"] in test]
    by = {}
    for o in rows:
        by.setdefault(test[o["prompt"]]["task"], []).append(o)
    out = {}
    for name, items in list(by.items()) + [("all", rows)]:
        n = len(items)
        acc = sum(o["wellformed"] == "1" and grade(o["prompt"], o["final"]) == 1.0 for o in items) / n
        out[name] = {"n": n, "wellformed": round(sum(o["wellformed"] == "1" for o in items) / n, 3),
                     "copy_err": round(sum(o["copy_error"] == "1" for o in items) / n, 3),
                     "final_transcribe_err": round(sum(o["wellformed"] == "1" and o["final"].strip() != o["result"]
                                                       for o in items) / n, 3),
                     "intent_or_tool_correct": round(sum(o["correct_tool"] == "1" for o in items) / n, 3),
                     "parse_reject": round(sum(o["note"].startswith("prompt rejected") for o in items) / n, 3),
                     "acc": round(acc, 3), "se": round((acc * (1 - acc) / n) ** 0.5, 3)}
    print(json.dumps({"tsv": path, **out}))
