#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic DSS dataset. Every answer is produced and re-verified by oracle.py.

Task formats (prompt -> answer, one line each):
  next  : "DSS greedy next after 3 5 6 ->" " 12"
  isdss : "is DSS 3 5 6 ->" " yes"  |  " no 6=1+5"   (witness: equal sums, sizes differ)
  card  : "DSS W 1 2 4 9 sum 11 -> size" " 2"
  bound : "bound k 5 a 16 ->" " yes 16>=13" | " no 4<5"   (a >= floor((k^2+1)/2))
Split: every example is keyed by its set (or (k,a)); keys are hashed into
train/test so no set appears in both. Test is a fixed held-out file.
"""
import hashlib, json, random, pathlib, sys
from oracle import DSSTable, is_dss, collision_witness, quadratic_bound

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
KERNEL = pathlib.Path("/workspace/sovereign-agi-kernel")
rng = random.Random(1234)


def key_split(key):
    h = int(hashlib.sha256(repr(key).encode()).hexdigest(), 16)
    return "test" if h % 10 == 0 else "train"


def rand_set(n, hi):
    return sorted(rng.sample(range(1, hi + 1), n))


def rand_dss(n, hi):
    for _ in range(1000):
        s = rand_set(n, hi)
        if is_dss(s):
            return s


def fmt(xs):
    return " ".join(map(str, xs))


def ex_next():
    s = rand_dss(rng.randint(2, 4), 30)
    t = DSSTable(s); x = s[-1] + 1
    while t.conflict(x) is not None:
        x += 1
    return ("next", tuple(s)), f"DSS greedy next after {fmt(s)} ->", f" {x}"


def ex_isdss():
    s = rand_dss(rng.randint(2, 5), 40) if rng.random() < 0.5 else rand_set(rng.randint(2, 5), 25)
    if is_dss(s):
        ans = " yes"
    else:
        a, b = collision_witness(s)
        if len(a) > len(b):
            a, b = b, a
        ans = f" no {'+'.join(map(str, a))}={'+'.join(map(str, b))}"
    return ("isdss", tuple(s)), f"is DSS {fmt(s)} ->", ans


def ex_card():
    s = rand_dss(rng.randint(2, 5), 40)
    t = DSSTable(s)
    sums = [i for i in range(1, len(t.card)) if t.card[i] >= 0]
    v = rng.choice(sums)
    return ("card", tuple(s)), f"DSS W {fmt(s)} sum {v} -> size", f" {t.cardinality(v)}"


def ex_bound():
    k = rng.randint(1, 30); b = quadratic_bound(k)
    a = max(1, b + rng.randint(-b // 2 - 1, b // 2 + 1))
    ans = f" yes {a}>={b}" if a >= b else f" no {a}<{b}"
    return ("bound", k, a), f"bound k {k} a {a} ->", ans


GEN = {"next": ex_next, "isdss": ex_isdss, "card": ex_card, "bound": ex_bound}


def main(n_per_task=6000, n_test_per_task=100):
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, seen = [], {t: [] for t in GEN}, set()
    for task, g in GEN.items():
        tries = 0
        while sum(1 for r in train if r["task"] == task) < n_per_task and tries < 20 * n_per_task:
            tries += 1
            key, p, a = g()
            if (key, p) in seen:
                continue
            seen.add((key, p))
            row = {"task": task, "prompt": p, "answer": a}
            if key_split(key) == "test":
                if len(test[task]) < n_test_per_task:
                    test[task].append(row)
            else:
                train.append(row)
    from grade import grade  # independent re-check of every stored answer
    allrows = train + [r for v in test.values() for r in v]
    assert all(grade(r["prompt"], r["answer"]) == 1.0 for r in allrows)
    with open(OUT / "train.jsonl", "w") as f:
        for r in train: f.write(json.dumps(r) + "\n")
    with open(OUT / "test.jsonl", "w") as f:
        for v in test.values():
            for r in v: f.write(json.dumps(r) + "\n")
    docs = [KERNEL / "README.md", KERNEL / "runtime/greedy.mjs"]
    with open(OUT / "kernel_text.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps({"path": str(d.relative_to(KERNEL)), "text": d.read_text()}) + "\n")
    stats = {"train": {t: sum(r["task"] == t for r in train) for t in GEN},
             "test": {t: len(v) for t, v in test.items()}, "kernel_docs": len(docs)}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
