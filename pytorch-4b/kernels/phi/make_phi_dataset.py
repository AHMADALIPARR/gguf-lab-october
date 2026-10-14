#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Phi-kernel intent data. Answers come from the Rust exact kernel
(dss-host answer), which was cross-checked against the Ruby PhiAlgebra.

Writes into pytorch-4b/dss/data/:
  phi_train.jsonl        2000 per task, |coef| <= 50 (25% up to 1000), k <= 12 (25% up to 20)
  phi_test.jsonl         100 per task, same distribution, disjoint prompts
  phi_test_ood.jsonl     100 per task, |coef| in 10^6..10^9, k up to 60
  test_mixed_balanced.jsonl = dss test_balanced + phi_test
  test_mixed_ood.jsonl      = dss test_ood + phi_test_ood
"""
import hashlib, json, random, subprocess, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phi_prompts import gen
DATA = HERE.parent.parent / "dss/data"
HOST = HERE.parent.parent / "dss/host/target/release/dss-host"
TASKS = ("phi_normalize", "phi_mul", "phi_conj", "phi_norm")
rng = random.Random(2026)


def split(p):
    return "test" if int(hashlib.sha256(p.encode()).hexdigest(), 16) % 10 == 0 else "train"


def answer(prompts):
    out = subprocess.run([str(HOST), "answer"], input="\n".join(prompts) + "\n", capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert not any(o.startswith("ERROR") for o in out), [o for o in out if o.startswith("ERROR")][:3]
    return out


def big(rng):
    m = rng.randint(10 ** 6, 10 ** 9)
    return m if rng.random() < 0.5 else -m


train, test, ood, seen = [], [], [], set()
for task in TASKS:
    ntr = nte = 0
    while ntr < 2000 or nte < 100:
        mag = 1000 if rng.random() < 0.25 else 50
        kmax = 20 if rng.random() < 0.25 else 12
        _, p = gen(rng, task, mag, kmax)
        if p in seen: continue
        seen.add(p)
        if split(p) == "test":
            if nte < 100: test.append({"task": task, "prompt": p}); nte += 1
        elif ntr < 2000:
            train.append({"task": task, "prompt": p}); ntr += 1
    n = 0
    while n < 100:
        q, _ = gen(rng, task, 1, 60)
        # replace coefficients with large magnitudes, keep structure
        from phi_prompts import poly_txt, elem_txt
        if task == "phi_normalize":
            terms = [(big(rng), rng.randint(0, 60)) for _ in q["terms"]]
            p = f"phi normalize {poly_txt(terms)} ->"
        elif task == "phi_mul":
            p = f"phi mul {elem_txt(big(rng), big(rng))} {elem_txt(big(rng), big(rng))} ->"
        else:
            p = f"phi {task.split('_')[1]} {elem_txt(big(rng), big(rng))} ->"
        if p in seen: continue
        seen.add(p); ood.append({"task": task, "prompt": p}); n += 1
for rows in (train, test, ood):
    for r, a in zip(rows, answer([r["prompt"] for r in rows])):
        r["answer"] = " " + a
rng.shuffle(train)


def dump(name, rows):
    with open(DATA / name, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


dump("phi_train.jsonl", train); dump("phi_test.jsonl", test); dump("phi_test_ood.jsonl", ood)
dss_bal = [json.loads(l) for l in open(DATA / "test_balanced.jsonl")]
dss_ood = [json.loads(l) for l in open(DATA / "test_ood.jsonl")]
dump("test_mixed_balanced.jsonl", dss_bal + test); dump("test_mixed_ood.jsonl", dss_ood + ood)
print({"phi_train": len(train), "phi_test": len(test), "phi_test_ood": len(ood),
       "mixed_balanced": len(dss_bal) + len(test), "mixed_ood": len(dss_ood) + len(ood)})
