#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Incremental verifier for distinct-subset-size-by-sum sets ("DSS" in
sovereign-agi-kernel runtime/greedy.mjs).

A set S of positive integers is DSS iff any two subsets of S with equal sum
have equal cardinality, i.e. the sum of a subset determines its size.

Instead of enumerating all 2^n subsets for every candidate (greedy.mjs), we
keep a table card[s] = size of a subset with sum s (or -1), built by a 0/1
knapsack-style update. Adding x to a DSS set is valid iff, for every
reachable sum s, card[s+x] is either unreachable or equal to card[s]+1.
That is O(sum(S)) per candidate with numpy.
"""
import numpy as np

NONE = -1


class DSSTable:
    def __init__(self, elems=()):
        self.elems = []
        self.card = np.array([0], dtype=np.int16)  # empty subset: sum 0, size 0
        for x in elems:
            if not self.add(x):
                raise ValueError(f"adding {x} breaks DSS")

    def conflict(self, x):
        """Return None if S+{x} stays DSS, else (sum, size_a, size_b)."""
        c = self.card
        n = len(c)
        reach = np.nonzero(c != NONE)[0]
        tgt = reach + x
        inside = tgt < n
        r, t = reach[inside], tgt[inside]
        bad = (c[t] != NONE) & (c[t] != c[r] + 1)
        if bad.any():
            i = int(np.argmax(bad))
            return int(t[i]), int(c[t[i]]), int(c[r[i]] + 1)
        return None

    def add(self, x):
        if x <= 0 or x in self.elems or self.conflict(x) is not None:
            return False
        c = self.card
        new = np.full(len(c) + x, NONE, dtype=np.int16)
        new[:len(c)] = c
        reach = np.nonzero(c != NONE)[0]
        new[reach + x] = c[reach] + 1
        self.card = new
        self.elems.append(x)
        return True

    def cardinality(self, s):
        return int(self.card[s]) if 0 <= s < len(self.card) else NONE


def is_dss(S):
    t = DSSTable()
    for x in S:
        if x <= 0 or x in t.elems or t.conflict(x) is not None:
            return False
        t.add(x)
    return True


def collision_witness(S):
    """For a non-DSS set, return two subsets (A, B) with equal sum, |A|!=|B|."""
    S = list(S)
    best = {0: ()}  # sum -> one subset
    for x in S:
        for s, sub in list(best.items()):
            ns, nsub = s + x, sub + (x,)
            if ns in best:
                if len(best[ns]) != len(nsub):
                    return best[ns], nsub
            else:
                best[ns] = nsub
    # fallback exhaustive (dict above keeps one representative per sum, which
    # suffices: any conflict is found when the second subset is formed)
    return None


def greedy(k, start=(1,)):
    t = DSSTable(start)
    seq = list(start)
    while len(seq) < k:
        x = seq[-1] + 1
        while t.conflict(x) is not None:
            x += 1
        t.add(x)
        seq.append(x)
    return seq


def quadratic_bound(k):
    """Bound checked by greedy.mjs verifyQuadraticBound: a_k >= floor((k^2+1)/2)."""
    return (k * k + 1) // 2


if __name__ == "__main__":
    import sys, time
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    t0 = time.time()
    print(greedy(k), f"{time.time()-t0:.2f}s")
