#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Large-magnitude data for the tool-calling rounds. All answers are checked by grade.py.

train_aug.jsonl : 8000 prompts (2000 per task) whose numbers have 1-5 digits,
                  mostly 3-5, so the model sees many BPE splits of long
                  numbers. Same prompt formats as train_v2.
test_ood.jsonl  : 100 per task with 3-5 digit numbers (is-DSS 50 yes / 50 no,
                  bound 50/50 with k in 45..300), plus the two prompts that
                  failed in the live llama.cpp test. Keys are hash-split as in
                  make_dataset.py, so no OOD test set or (k, a) pair is in
                  train_aug.
Non-DSS sets are made by planting a collision (an element equal to the sum
of two others), because random large sets are almost always DSS.
"""
import json, random, collections
from make_dataset import key_split
from oracle import DSSTable, is_dss, quadratic_bound
from make_dataset_v2 import isdss_row, bound_row
from grade import grade

rng = random.Random(4242)


def num(lo=3, hi=5):
    d = rng.randint(lo, hi)
    return rng.randint(10 ** (d - 1), 10 ** d - 1)


def mixed():
    return num(1, 5) if rng.random() < 0.25 else num()


def big_dss(n, f):
    while True:
        s = sorted({f() for _ in range(n)})
        if len(s) == n and is_dss(s): return s


def gen(task, label, f, big):
    if task == "isdss":
        n = rng.randint(2, 4)
        if label:
            s = big_dss(n, f)
        else:
            while True:
                a, b = f(), f()
                s = sorted({a, b, a + b, *[f() for _ in range(max(0, n - 3))]})
                if len(s) >= 3 and not is_dss(s): break
        return ("isdss", tuple(s)), isdss_row(s)
    if task in ("card", "next"):
        s = big_dss(rng.randint(2, 4), f); t = DSSTable(s)
        if task == "card":
            v = sum(rng.sample(s, rng.randint(1, len(s))))
            return ("card", tuple(s)), {"task": "card", "prompt": f"DSS W {' '.join(map(str, s))} sum {v} -> size",
                                        "answer": f" {t.cardinality(v)}"}
        x = s[-1] + 1
        while t.conflict(x) is not None: x += 1
        return ("next", tuple(s)), {"task": "next", "prompt": f"DSS greedy next after {' '.join(map(str, s))} ->",
                                    "answer": f" {x}"}
    k = rng.randint(45, 300) if big else rng.randint(1, 300)
    b = quadratic_bound(k); span = max(1, b // rng.choice([1, 2, 4, 10, 100]))
    a = b + rng.randint(0, span) if label else max(1, b - rng.randint(1, span))
    return ("bound", k, a), bound_row(k, a)


def main():
    seen, train, test = set(), [], []
    for task in ("isdss", "card", "next", "bound"):
        tr, te = 0, {True: 0, False: 0}
        need = 50 if task in ("isdss", "bound") else 100
        i = 0
        while tr < 2000 or sum(te.values()) < 100:
            i += 1
            label = i % 2 == 0 if task in ("isdss", "bound") else True
            key, row = gen(task, label, num if rng.random() < 0.5 else mixed, rng.random() < 0.5)
            if row["prompt"] in seen: continue
            if key_split(key) == "test":
                ints = [int(x) for x in row["prompt"].split() if x.isdigit()]
                ok = (ints[0] >= 45 and ints[1] >= 100) if task == "bound" else min(ints) >= 100
                if ok and te[label] < need:
                    test.append(row); te[label] += 1; seen.add(row["prompt"])
            elif tr < 2000:
                train.append(row); tr += 1; seen.add(row["prompt"])
    test.append({**bound_row(25, 313), "src": "live-failure"})
    test.append({**isdss_row([101, 202, 303]), "src": "live-failure"})
    for r in train + test:
        assert grade(r["prompt"], r["answer"]) == 1.0, r
    assert not ({r["prompt"] for r in train} & {r["prompt"] for r in test})
    rng.shuffle(train)
    with open("data/train_aug.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open("data/test_ood.jsonl", "w") as f:
        for r in test: f.write(json.dumps(r) + "\n")
    st = {"train_aug": dict(collections.Counter(r["task"] for r in train)),
          "test_ood": dict(collections.Counter(r["task"] for r in test)),
          "ood_isdss_yes": sum(r["task"] == "isdss" and r["answer"] == " yes" for r in test),
          "ood_bound_yes": sum(r["task"] == "bound" and r["answer"].endswith("yes") for r in test)}
    json.dump(st, open("data/stats_ood.json", "w"), indent=1); print(st)


if __name__ == "__main__":
    main()
