#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Prompt generators for the phi kernel tasks (shared by the dataset builder
and the Ruby cross-check)."""


def elem_txt(a, b):
    return f"({a} + {b} phi)" if b >= 0 else f"({a} - {-b} phi)"


def poly_txt(terms):
    out = []
    for i, (c, k) in enumerate(terms):
        t = str(c) if i == 0 else str(abs(c))
        if k: t += f" phi^{k}"
        if i: out.append("+" if c >= 0 else "-")
        out.append(t)
    return " ".join(out)


def rand_int(rng, mag):
    return rng.randint(-mag, mag)


def gen(rng, task, mag, kmax):
    if task == "phi_normalize":
        n = rng.randint(1, 4)
        terms = [(rand_int(rng, mag) or 1, rng.randint(0, kmax)) for _ in range(n)]
        return {"op": "normalize", "terms": terms}, f"phi normalize {poly_txt(terms)} ->"
    x = (rand_int(rng, mag), rand_int(rng, mag))
    if task == "phi_mul":
        y = (rand_int(rng, mag), rand_int(rng, mag))
        return {"op": "mul", "x": x, "y": y}, f"phi mul {elem_txt(*x)} {elem_txt(*y)} ->"
    op = task.split("_")[1]
    return {"op": op, "x": x}, f"phi {op} {elem_txt(*x)} ->"
