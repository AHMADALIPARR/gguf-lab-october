#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-check the Rust kernel (host/, `dss-host kernel`) against oracle.py
(via agent.run_tool) and runtime/greedy.mjs (node) on every tool query of the
balanced, round-1 and OOD test sets plus 2000 training prompts."""
import json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from agent import expected_call, run_tool

KERNEL = "/workspace/sovereign-agi-kernel/runtime/greedy.mjs"
rows = []
for f in ["test_balanced.jsonl", "test.jsonl", "test_ood.jsonl"]:
    rows += [json.loads(l) for l in open(HERE / "data" / f)]
rows += [json.loads(l) for l in open(HERE / "data/train_v2.jsonl")][:1000]
rows += [json.loads(l) for l in open(HERE / "data/train_aug.jsonl")][:1000]
q = [expected_call(r["prompt"]) for r in rows]
lines = [f"{n} {a}" for n, a in q]
rust = subprocess.run([str(HERE / "host/target/release/dss-host"), "kernel"], input="\n".join(lines) + "\n",
                      capture_output=True, text=True, check=True).stdout.splitlines()
py = [run_tool(n, a) for n, a in q]
js_src = f"""
const g = await import('file://{KERNEL}');
const qs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
""".replace("require('fs')", "(await import('fs')).default")
js_src += r"""
const out = qs.map(([n, a]) => {
  const xs = a.split(' ');
  if (n === 'isdss') return g.isDSS(xs.map(Number)) ? 'yes' : 'no';
  if (n === 'card') { const i = xs.indexOf('sum'); return String(g.cardinalityFromSum(xs.slice(0, i).map(Number), Number(xs[i + 1]))); }
  if (n === 'next') { const s = xs.map(Number); let x = s[s.length - 1] + 1; while (!g.isDSS([...s, x])) x++; return String(x); }
  if (n === 'bound') { const k = Number(xs[0]), a2 = Number(xs[1]); const r = g.verifyQuadraticBound(Array(k).fill(a2))[k - 1];
    return `b=${r.bound}; ${a2}${r.holds ? '>=' : '<'}${r.bound} ${r.holds ? 'yes' : 'no'}`; }
});
console.log(JSON.stringify(out));
"""
js = json.loads(subprocess.run(["node", "--input-type=module", "-e", js_src], input=json.dumps(q),
                               capture_output=True, text=True, check=True).stdout)
res = {}
for (n, _), r, p, j in zip(q, rust, py, js):
    d = res.setdefault(n, {"total": 0, "rust_eq_python": 0, "rust_eq_greedy_mjs": 0})
    d["total"] += 1
    d["rust_eq_python"] += r == p
    # greedy.mjs isDSS returns no witness, so compare the verdict only
    d["rust_eq_greedy_mjs"] += (r.split()[0] == j) if n == "isdss" else (r == j)
print(json.dumps(res, indent=1))
