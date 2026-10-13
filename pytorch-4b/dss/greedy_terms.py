#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Compute greedy DSS terms with the incremental oracle, logging time per term."""
import json, sys, time
from oracle import DSSTable, quadratic_bound
K = int(sys.argv[1]); out = sys.argv[2]
t = DSSTable([1]); seq = [1]; rows = [[1, 1, 0.0]]; t0 = time.time()
while len(seq) < K:
    x = seq[-1] + 1
    while t.conflict(x) is not None:
        x += 1
    t.add(x); seq.append(x); rows.append([len(seq), x, round(time.time() - t0, 2)])
    print(rows[-1], flush=True)
json.dump({"terms": seq, "rows_k_term_elapsed_s": rows,
           "bound_fails_at_k": [k for k, a in enumerate(seq, 1) if a < quadratic_bound(k)]},
          open(out, "w"), indent=1)
