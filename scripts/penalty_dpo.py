#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Short reward-weighted DPO on /workspace/neg-rl/pairs.jsonl.
Reward per item (from pairs.jsonl, per neg-rl README): score in {-1,0,+1} plus
min(0.5*novelty,0.5) only when score==+1. Pair weight = (chosen_reward - rejected_reward)/2.
Loss = w * -logsigmoid(beta * ((pc - rc) - (pr - rr))). Reference logprobs precomputed."""
import argparse, json, random, sys, time, math, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/gguf-lab-october/pytorch-4b")
from eval import load_model
from tokenizers import Tokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pairs", default="/workspace/neg-rl/pairs.jsonl")
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--seq", type=int, default=256)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0); random.seed(0)
tok = Tokenizer.from_file("/workspace/gguf-lab-october/pytorch-4b/tokenizer/tokenizer.json")
model, cfg, ck = load_model(a.ckpt)
def enc(p, c):
    pi = tok.encode(p).ids[: a.seq // 2]; ci = tok.encode(c).ids
    ids = (pi + ci)[: a.seq]; return ids, len(pi)
def logp(m, ids, np_):
    x = torch.tensor([ids]); lg = m(x[:, :-1]).float().log_softmax(-1)
    t = x[:, 1:]; lp = lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)[0]
    return lp[np_ - 1:].sum()
P = []
for l in open(a.pairs):
    r = json.loads(l)
    c, cn = enc(r["prompt"], r["chosen"]); j, jn = enc(r["prompt"], r["rejected"])
    if len(c) <= cn or len(j) <= jn: continue
    P.append(dict(c=c, cn=cn, j=j, jn=jn, w=(r["chosen_reward"] - r["rejected_reward"]) / 2))
model.eval(); t0 = time.time()
with torch.no_grad():
    for p in P: p["rc"] = float(logp(model, p["c"], p["cn"])); p["rj"] = float(logp(model, p["j"], p["jn"]))
print(f"pairs={len(P)} ref logprobs {time.time()-t0:.0f}s", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
steps = a.epochs * math.ceil(len(P) / a.batch); step = 0; log = open(a.out + ".log.jsonl", "w")
model.train()
for ep in range(a.epochs):
    random.shuffle(P)
    for i in range(0, len(P), a.batch):
        B = P[i:i + a.batch]; L = 0; acc = 0; mg = 0
        for p in B:
            d = (logp(model, p["c"], p["cn"]) - p["rc"]) - (logp(model, p["j"], p["jn"]) - p["rj"])
            loss = p["w"] * -F.logsigmoid(a.beta * d) / len(B); loss.backward()
            L += float(loss); acc += float(d > 0) / len(B); mg += float(d) / len(B)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        rec = dict(step=step, steps=steps, loss=round(L, 4), pref_acc=round(acc, 3), margin=round(mg, 3), s=round(time.time() - t0))
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
model.eval()
# final preference accuracy over all pairs (policy vs ref)
with torch.no_grad():
    acc = sum(float((logp(model, p["c"], p["cn"]) - p["rc"]) - (logp(model, p["j"], p["jn"]) - p["rj"]) > 0) for p in P) / len(P)
    raw = sum(float(logp(model, p["c"], p["cn"]) > logp(model, p["j"], p["jn"])) for p in P) / len(P)
    raw_ref = sum(float(p["rc"] > p["rj"]) for p in P) / len(P)
print(json.dumps(dict(final_implicit_pref_acc=round(acc, 4), raw_logp_chosen_gt_rejected_policy=round(raw, 4), raw_ref=round(raw_ref, 4))), flush=True)
torch.save({"model": model.state_dict(), "config": ck["config"]}, a.out)
print("saved", a.out)
