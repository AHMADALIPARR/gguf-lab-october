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


# Phi-kernel prompts are graded against the Rust exact kernel's answers
# (kernels/phi; cross-checked against the Ruby PhiAlgebra), registered here
# from the dataset rows by agent.register_phi().
PHI_ANS = {}


def grade(prompt, completion):
    ans = completion.split("\n")[0].strip()
    if (prompt.startswith("phi ") or prompt.startswith("viterbi ")
            or prompt.startswith("permission ") or prompt.startswith("governance ")
            or prompt.startswith("worm seal ")):
        if prompt not in PHI_ANS:
            raise ValueError(f"no kernel answer registered for {prompt}")
        return float(ans == PHI_ANS[prompt])
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
            # v3 stepwise forms: "<certificate> yes" (certificate not graded)
            # and "A=B no" (witness first). Only the verdict and, for "no",
            # the witness are graded.
            if ans == "yes" or (ans.endswith(" yes") and ans.startswith("sums ")):
                return float(d)
            w3 = re.fullmatch(r"(\d+(?:\+\d+)*)=(\d+(?:\+\d+)*) no", ans)
            if w3:
                ans = f"no {w3.group(1)}={w3.group(2)}"
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
            # v1 format "yes a>=b" / "no a<b"; v2 format "b=B; a>=B yes" /
            # "b=B; a<B no" (bound stated before the verdict). Either is
            # accepted, but every number and the verdict must be correct.
            v1 = f"yes {a}>={b}" if a >= b else f"no {a}<{b}"
            v2 = f"b={b}; {a}>={b} yes" if a >= b else f"b={b}; {a}<{b} no"
            if not re.fullmatch(r"(yes \d+>=\d+|no \d+<\d+|b=\d+; \d+(>=|<)\d+ (yes|no))", ans):
                return MALFORMED
            return float(ans in (v1, v2))
    except (ValueError, OverflowError):
        return MALFORMED
    raise ValueError(f"unknown prompt: {prompt}")
