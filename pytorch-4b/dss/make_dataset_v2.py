#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Round-2 data: bound task in a stepwise format, balanced is-DSS.

Writes (in data/):
  train_v2.jsonl       old next/card/isdss train rows + extra non-DSS isdss
                       rows (to ~50/50) + 10000 bound rows (k in 1..40) in the v2 format
                       "bound k 16 a 67 -> b=128; 67<128 no", balanced
                       yes/no, with a concentrated near the bound.
  test_balanced.jsonl  next/card from test.jsonl, 100 yes + 100 no is-DSS
                       sets, 100 balanced v2 bound items. Keys use the same
                       hash split as make_dataset.py, so none are in train.
test.jsonl (round 1) is unchanged.
"""
import json, random
from make_dataset import key_split, rand_set, rand_dss, fmt
from oracle import is_dss, collision_witness, quadratic_bound
from grade import grade

rng = random.Random(777)
import make_dataset; make_dataset.rng = rng


def isdss_row(s):
    if is_dss(s):
        return {"task": "isdss", "prompt": f"is DSS {fmt(s)} ->", "answer": " yes"}
    a, b = collision_witness(s)
    if len(a) > len(b): a, b = b, a
    return {"task": "isdss", "prompt": f"is DSS {fmt(s)} ->",
            "answer": f" no {'+'.join(map(str, a))}={'+'.join(map(str, b))}"}


def bound_row(k, a):
    b = quadratic_bound(k)
    ans = f" b={b}; {a}>={b} yes" if a >= b else f" b={b}; {a}<{b} no"
    return {"task": "bound", "prompt": f"bound k {k} a {a} ->", "answer": ans}


def rand_bound(want_yes):
    k = rng.randint(1, 40); b = quadratic_bound(k)
    span = max(1, b // rng.choice([1, 2, 4, 10]))
    if want_yes:
        a = b + rng.randint(0, span)
    else:
        if b == 1: return None
        a = max(1, b - rng.randint(1, span))
    return ("bound", k, a), (k, a)


def main():
    old_train = [json.loads(l) for l in open("data/train.jsonl")]
    old_test = [json.loads(l) for l in open("data/test.jsonl")]
    train = [r for r in old_train if r["task"] != "bound"]
    seen = {r["prompt"] for r in old_train + old_test}
    # balanced bound
    btrain, btest = [], {True: [], False: []}
    i = 0
    while len(btrain) < 10000 or min(len(v) for v in btest.values()) < 50:
        want = i % 2 == 0; i += 1
        g = rand_bound(want)
        if g is None: continue
        key, (k, a) = g
        row = bound_row(k, a)
        if row["prompt"] in seen: continue
        seen.add(row["prompt"])
        if key_split(key) == "test":
            if len(btest[want]) < 50: btest[want].append(row)
        elif len(btrain) < 10000:
            btrain.append(row)
    train += btrain
    # extra non-DSS train rows + balanced is-DSS test
    no_train, itest = [], {True: [], False: []}
    while len(no_train) < 4000 or min(len(v) for v in itest.values()) < 100:
        if rng.random() < 0.5:
            s = rand_set(rng.randint(2, 5), 25)
        else:
            s = rand_dss(rng.randint(2, 5), 40)
        d = is_dss(s); row = isdss_row(s)
        if row["prompt"] in seen: continue
        if key_split(("isdss", tuple(s))) == "test":
            if len(itest[d]) < 100:
                itest[d].append(row); seen.add(row["prompt"])
        elif not d and len(no_train) < 4000:
            no_train.append(row); seen.add(row["prompt"])
    train += no_train
    rng.shuffle(train)
    test = [r for r in old_test if r["task"] in ("next", "card")] + itest[True] + itest[False] \
        + btest[True] + btest[False]
    for r in train + test:
        assert grade(r["prompt"], r["answer"]) == 1.0, r
    with open("data/train_v2.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open("data/test_balanced.jsonl", "w") as f:
        for r in test: f.write(json.dumps(r) + "\n")
    import collections
    st = {"train_v2": dict(collections.Counter(r["task"] for r in train)),
          "train_v2_isdss_yes_no": dict(collections.Counter(r["answer"].split()[0] for r in train if r["task"] == "isdss")),
          "train_v2_bound_yes_no": dict(collections.Counter(r["answer"].split()[-1] for r in train if r["task"] == "bound")),
          "test_balanced": dict(collections.Counter(r["task"] for r in test)),
          "test_balanced_isdss_yes_no": {"yes": 100, "no": 100}, "test_balanced_bound_yes_no": {"yes": 50, "no": 50}}
    json.dump(st, open("data/stats_v2.json", "w"), indent=1); print(st)


if __name__ == "__main__":
    main()
