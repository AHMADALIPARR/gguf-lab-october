#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic grader: reward 1.0 verified correct, 0.0 wrong, -0.1 malformed.

Graded only from the prompt and the oracle; never against a stored answer, so
any correct witness for a non-DSS set is accepted."""
import re
from collections import Counter
from oracle import DSSTable, is_dss, quadratic_bound

MALFORMED = -0.1


def _ints(s):
    return [int(x) for x in s.split()]


def _multisubset(part, S):
    c, have = Counter(part), Counter(S)
    return all(have[x] >= n for x, n in c.items())


def grade(prompt, completion):
    ans = completion.split("\n")[0].strip()
    try:
        m = re.fullmatch(r"DSS greedy next after ([\d ]+) ->", prompt)
        if m:
            if not re.fullmatch(r"\d+", ans): return MALFORMED
            S = _ints(m.group(1)); t = DSSTable(S); x = S[-1] + 1
            while t.conflict(x) is not None: x += 1
            return float(int(ans) == x)
        m = re.fullmatch(r"is DSS ([\d ]+) ->", prompt)
        if m:
            S = _ints(m.group(1)); d = is_dss(S)
            if ans == "yes": return float(d)
            w = re.fullmatch(r"no (\d+(?:\+\d+)*)=(\d+(?:\+\d+)*)", ans)
            if not w: return MALFORMED
            a = [int(x) for x in w.group(1).split("+")]; b = [int(x) for x in w.group(2).split("+")]
            ok = (not d and sum(a) == sum(b) and len(a) != len(b) and len(set(a)) == len(a)
                  and len(set(b)) == len(b) and _multisubset(a, S) and _multisubset(b, S))
            return float(ok)
        m = re.fullmatch(r"DSS W ([\d ]+) sum (\d+) -> size", prompt)
        if m:
            if not re.fullmatch(r"\d+", ans): return MALFORMED
            t = DSSTable(_ints(m.group(1)))
            return float(int(ans) == t.cardinality(int(m.group(2))))
        m = re.fullmatch(r"bound k (\d+) a (\d+) ->", prompt)
        if m:
            k, a = int(m.group(1)), int(m.group(2)); b = quadratic_bound(k)
            exp = f"yes {a}>={b}" if a >= b else f"no {a}<{b}"
            if not re.fullmatch(r"(yes \d+>=\d+|no \d+<\d+)", ans): return MALFORMED
            return float(ans == exp)
    except (ValueError, OverflowError):
        return MALFORMED
    raise ValueError(f"unknown prompt: {prompt}")
