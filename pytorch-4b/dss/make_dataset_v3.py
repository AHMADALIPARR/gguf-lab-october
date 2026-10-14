#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Round-3 data: is-DSS answers with the reasoning written out.

train_v3.jsonl = train_v2.jsonl with every is-DSS answer rewritten:
  non-DSS: " 6=1+5 no"   two colliding subsets (equal sum, different size)
           from the oracle, smaller subset first, then the verdict.
  DSS:     " sums 1:[3 5 7] 2:[8 10 12] 3:[15] yes"  subset sums grouped by
           subset size, each group sorted. No sum appears in two groups,
           which is the DSS property. Sets have 2..5 elements, so at most 31
           sums (about 60 tokens).
The grader scores only the verdict and, for "no", the witness.
test_balanced.jsonl is reused unchanged (it is graded from the prompt).
"""
import json, re
from itertools import combinations
from oracle import is_dss, collision_witness
from grade import grade


def certificate(S):
    groups = []
    for r in range(1, len(S) + 1):
        groups.append(f"{r}:[{' '.join(map(str, sorted({sum(c) for c in combinations(S, r)})))}]")
    return "sums " + " ".join(groups)


def answer(S):
    if is_dss(S):
        return f" {certificate(S)} yes"
    a, b = collision_witness(S)
    if len(a) > len(b): a, b = b, a
    return f" {'+'.join(map(str, a))}={'+'.join(map(str, b))} no"


def main():
    out, n = [], {"yes": 0, "no": 0}
    for l in open("data/train_v2.jsonl"):
        r = json.loads(l)
        if r["task"] == "isdss":
            S = [int(x) for x in re.fullmatch(r"is DSS ([\d ]+) ->", r["prompt"]).group(1).split()]
            r["answer"] = answer(S); n[r["answer"].split()[-1]] += 1
        assert grade(r["prompt"], r["answer"]) == 1.0, r
        out.append(r)
    with open("data/train_v3.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    json.dump({"train_v3": len(out), "isdss_yes_no": n}, open("data/stats_v3.json", "w"), indent=1)
    print(len(out), n)


if __name__ == "__main__":
    main()
