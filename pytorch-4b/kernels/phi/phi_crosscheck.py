#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Rust exact phi kernel (dss-host answer) vs sovereign-ruby-kernel PhiAlgebra (ruby, f64).

The Rust result a + b phi (or an integer norm) is evaluated to 50 significant
digits with Decimal, then compared with Ruby's f64. Agreement means
|ruby - exact| <= 1e-9 * scale, where scale is the sum of absolute values of
the terms Ruby adds (the f64 rounding budget). Reported separately for small
(|coef| <= 50, k <= 12) and large (|coef| up to 10^9, k <= 60) inputs."""
import json, random, subprocess, sys, pathlib
from decimal import Decimal, getcontext
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phi_prompts import gen
getcontext().prec = 60
PHI = (1 + Decimal(5).sqrt()) / 2
HOST = HERE.parent.parent / "dss/host/target/release/dss-host"
RUBY_KERNEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sovereign-ruby-kernel"


def val(s):
    t = s.split()
    if len(t) == 1: return Decimal(t[0])
    a, sign, b = Decimal(t[0]), t[1], Decimal(t[2])
    return a + (b if sign == "+" else -b) * PHI


def scale(q):
    if q["op"] == "normalize": return sum(abs(Decimal(c)) * PHI ** k for c, k in q["terms"]) + 1
    x = abs(Decimal(q["x"][0])) + abs(Decimal(q["x"][1])) * PHI
    if q["op"] == "mul": return x * (abs(Decimal(q["y"][0])) + abs(Decimal(q["y"][1])) * PHI) + 1
    if q["op"] == "norm": return x * x + 1
    return x + 1


rng = random.Random(7)
res = {}
for label, mag, kmax in [("small", 50, 12), ("large", 10 ** 9, 60)]:
    qs, prompts, tasks = [], [], []
    for task in ("phi_normalize", "phi_mul", "phi_conj", "phi_norm"):
        for _ in range(500):
            q, p = gen(rng, task, mag, kmax); qs.append(q); prompts.append(p); tasks.append(task)
    rust = subprocess.run([str(HOST), "answer"], input="\n".join(prompts) + "\n", capture_output=True,
                          text=True, check=True).stdout.splitlines()
    ruby = json.loads(subprocess.run(["ruby", str(HERE / "ruby_eval.rb"), RUBY_KERNEL], input=json.dumps(qs),
                                     capture_output=True, text=True, check=True).stdout)
    for t, q, r, f in zip(tasks, qs, rust, ruby):
        d = res.setdefault(f"{label}/{t}", {"total": 0, "agree": 0, "rust_error": 0, "max_rel_err": 0.0})
        d["total"] += 1
        if r.startswith("ERROR"):
            d["rust_error"] += 1; continue
        err = abs(Decimal(f) - val(r)) / scale(q)
        d["max_rel_err"] = max(d["max_rel_err"], float(err))
        d["agree"] += err <= Decimal("1e-9")
print(json.dumps(res, indent=1))

# Exactness of Ruby's f64 norm: does rounding Ruby's value recover the exact integer norm?
rng = random.Random(11)
for label, mag in [("small", 50), ("large", 10 ** 9)]:
    qs, ps = [], []
    for _ in range(500):
        q, p = gen(rng, "phi_norm", mag, 0); qs.append(q); ps.append(p)
    rust = subprocess.run([str(HOST), "answer"], input="\n".join(ps) + "\n", capture_output=True, text=True).stdout.split()
    ruby = json.loads(subprocess.run(["ruby", str(HERE / "ruby_eval.rb"), RUBY_KERNEL], input=json.dumps(qs),
                                     capture_output=True, text=True).stdout)
    hit = sum(int(Decimal(f).to_integral_value()) == int(r) for f, r in zip(ruby, rust))
    print(json.dumps({f"{label}/norm_exact_integer_from_ruby_round": hit, "total": len(qs)}))
