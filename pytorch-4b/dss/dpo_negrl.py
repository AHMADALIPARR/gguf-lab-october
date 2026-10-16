#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Short DPO on /workspace/neg-rl/pairs.jsonl (verifier-gated chosen/rejected).
Usage: dpo_negrl.py CKPT OUT.pt [--steps N] [--lr LR] [--beta B] [--batch B] [--threads T]
Reports pair preference accuracy / mean margin on a 15% held-out split before and after."""
import argparse, copy, json, random, sys, time, pathlib
import torch, torch.nn.functional as F
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval import load_model
from tokenizers import Tokenizer
ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--steps", type=int, default=120); ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pairs", default="/workspace/neg-rl/pairs.jsonl")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = random.Random(a.seed)
tok = Tokenizer.from_file(str(HERE.parent / "tokenizer/tokenizer.json"))
model, cfg, _ = load_model(a.ckpt); L = cfg.max_seq_len
ref = copy.deepcopy(model).eval()
for p in ref.parameters(): p.requires_grad_(False)
rows = [json.loads(l) for l in open(a.pairs)]
rng.shuffle(rows); nv = max(1, int(0.15 * len(rows))); val, train = rows[:nv], rows[nv:]
def enc(prompt, comp):
    p = tok.encode(prompt).ids; c = tok.encode(comp).ids + [2]
    ids = (p + c)[:L]; m = ([0] * len(p) + [1] * len(c))[:L]
    if sum(m[1:]) == 0:  # prompt alone fills context: keep tail of prompt
        p = p[-(L // 2):]; ids = (p + c)[:L]; m = ([0] * len(p) + [1] * len(c))[:L]
    return ids, m
def batch(items):
    T = max(len(i) for i, _ in items)
    x = torch.full((len(items), T), 2, dtype=torch.long); mk = torch.zeros(len(items), T)
    for j, (i, m) in enumerate(items):
        x[j, :len(i)] = torch.tensor(i); mk[j, :len(m)] = torch.tensor(m, dtype=torch.float)
    return x, mk
def seqlp(mdl, x, mk):
    lg = mdl(x[:, :-1]).float(); lp = torch.log_softmax(lg, -1).gather(-1, x[:, 1:, None])[..., 0]
    return (lp * mk[:, 1:]).sum(-1)
def dpo(mdl, rs):
    xc, mc = batch([enc(r["prompt"], r["chosen"]) for r in rs]); xr, mr = batch([enc(r["prompt"], r["rejected"]) for r in rs])
    pc, pr = seqlp(mdl, xc, mc), seqlp(mdl, xr, mr)
    with torch.no_grad(): rc, rr = seqlp(ref, xc, mc), seqlp(ref, xr, mr)
    z = a.beta * ((pc - rc) - (pr - rr))
    return -F.logsigmoid(z).mean(), (pc - pr).detach()
@torch.no_grad()
def evalp(mdl):
    ms = []
    for i in range(0, len(val), 8): ms += dpo(mdl, val[i:i + 8])[1].tolist()
    return {"n": len(ms), "pref_acc": round(sum(m > 0 for m in ms) / len(ms), 4), "mean_logp_margin": round(sum(ms) / len(ms), 4)}
print(json.dumps({"stage": "pre", "val": evalp(model), "n_train": len(train)}), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0); model.train(); t0 = time.time(); tot = 0
for s in range(1, a.steps + 1):
    loss, _ = dpo(model, rng.sample(train, a.batch)); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); tot += loss.item()
    if s % 20 == 0: print(f"[dpo] step {s}/{a.steps} loss {tot/20:.4f} {time.time()-t0:.0f}s", flush=True); tot = 0
model.eval()
print(json.dumps({"stage": "post", "val": evalp(model), "args": vars(a)}), flush=True)
torch.save({"model": model.state_dict(), "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}}, a.out)
print("saved", a.out)
