#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-check oracle.py against sovereign-agi-kernel runtime/greedy.mjs via node."""
import json, random, subprocess, sys, pathlib
from oracle import is_dss, greedy, quadratic_bound, DSSTable, collision_witness

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sovereign-agi-kernel/runtime/greedy.mjs"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 14
rng = random.Random(0)
sets = []
for _ in range(2000):
    n = rng.randint(1, 9)
    sets.append(sorted(rng.sample(range(1, rng.choice([20, 60, 300])), n)))
ref = json.loads(subprocess.check_output(
    ["node", str(pathlib.Path(__file__).with_name("crosscheck.mjs")),
     "file://" + KERNEL, json.dumps(sets), str(K)]))
assert ref["greedy"] == greedy(K), (ref["greedy"], greedy(K))
assert [[k, quadratic_bound(k), g >= quadratic_bound(k)] for k, g in
        enumerate(greedy(K), 1)] == ref["bound"]
mism = sum(is_dss(s) != r for s, r in zip(sets, ref["dss"]))
assert mism == 0, mism
for s, r in zip(sets, ref["dss"]):
    if not r:
        a, b = collision_witness(s)
        assert sum(a) == sum(b) and len(a) != len(b)
ntab = 0
for s, tab, r in zip(sets, ref["tables"], ref["dss"]):
    if r:  # for DSS sets the table is well defined; compare every entry
        t = DSSTable(s)
        assert all(t.cardinality(k) == v for k, v in tab), s
        ntab += 1
print(f"OK: greedy k={K} matches, {len(sets)} sets isDSS agree "
      f"({sum(ref['dss'])} DSS), {ntab} cardinality tables agree")
