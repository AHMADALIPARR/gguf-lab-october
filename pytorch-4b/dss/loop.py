#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Oracle feedback loop for the small CPU model.

Stages (all rewards come from grade.py, a deterministic verifier):
  0. SFT warm start on oracle-verified train.jsonl + kernel text, with replay
     windows from the original token stream to limit forgetting.
  1. Expert iteration: sample G answers per train-style prompt, keep only the
     samples the oracle verifies, add them to the pool, fine-tune; repeat.
  2. RL with verifiable rewards: GRPO-style group-relative advantages,
     reward 1 / 0 / -0.1 (correct / wrong / malformed), plus a per-token KL
     penalty (k3 estimator) against the frozen pre-RL reference model.
After each round: greedy-decoding oracle accuracy on the fixed test.jsonl and
next-token loss on a fixed slice of the old val split. Results -> metrics.jsonl.
"""
import argparse, copy, json, pathlib, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval import load_model  # noqa: E402
from grade import grade  # noqa: E402

EOS = 2


def lines(p):
    return [json.loads(l) for l in open(p)]


class Ctx:
    def __init__(self, a):
        from tokenizers import Tokenizer
        self.a = a
        self.tok = Tokenizer.from_file(str(HERE.parent / "tokenizer/tokenizer.json"))
        self.nl = self.tok.encode("\n").ids
        self.train = lines(HERE / "data/train.jsonl")
        self.test = lines(HERE / "data/test.jsonl")
        self.kernel_ids = [self.tok.encode(d["text"]).ids for d in lines(HERE / "data/kernel_text.jsonl")]
        self.corpus = np.memmap(HERE.parent / "data/tokens.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(HERE.parent / "data/val.bin", dtype=np.uint16, mode="r")
        self.rng = random.Random(a.seed)
        self.out = pathlib.Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        self.metrics = open(self.out / "metrics.jsonl", "a")

    def enc(self, s):
        return self.tok.encode(s).ids

    def log(self, **kw):
        kw["time"] = time.strftime("%H:%M:%S")
        print(json.dumps(kw), flush=True)
        self.metrics.write(json.dumps(kw) + "\n"); self.metrics.flush()


def example_tensors(ctx, rows):
    """Right-padded batch; loss only on answer tokens (+ trailing newline)."""
    seqs, masks = [], []
    for r in rows:
        p = ctx.enc(r["prompt"]); a = ctx.enc(r["answer"]) + ctx.nl
        seqs.append(p + a); masks.append([0] * len(p) + [1] * len(a))
    T = max(map(len, seqs))
    x = torch.zeros(len(seqs), T, dtype=torch.long); m = torch.zeros(len(seqs), T)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(k)] = torch.tensor(k, dtype=torch.float)
    return x, m


def masked_ce(model, x, m):
    logits = model(x[:, :-1])
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
    mm = m[:, 1:].reshape(-1)
    return (ce * mm).sum() / mm.sum()


def replay_batch(ctx, n, L):
    xs = []
    for _ in range(n):
        if ctx.rng.random() < 0.3:
            ids = ctx.rng.choice(ctx.kernel_ids)
            s = ctx.rng.randrange(max(1, len(ids) - L - 1)); w = ids[s:s + L + 1]
            w = w + [EOS] * (L + 1 - len(w))
        else:
            s = ctx.rng.randrange(len(ctx.corpus) - L - 1)
            w = ctx.corpus[s:s + L + 1].astype("int64").tolist()
        xs.append(w)
    x = torch.tensor(xs)
    return x


def sft(ctx, model, pool, steps, lr, tag):
    a = ctx.a
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train(); t0 = time.time(); tot = 0.0
    for step in range(1, steps + 1):
        x, m = example_tensors(ctx, ctx.rng.sample(pool, a.batch))
        loss = masked_ce(model, x, m)
        r = replay_batch(ctx, a.replay, a.replay_len)
        lg = model(r[:, :-1])
        rl = F.cross_entropy(lg.reshape(-1, lg.size(-1)), r[:, 1:].reshape(-1))
        (loss + a.replay_weight * rl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); tot += loss.item()
        if step % 25 == 0:
            print(f"[{tag}] step {step}/{steps} task_loss {tot/25:.4f} replay {rl.item():.3f} "
                  f"{time.time()-t0:.0f}s", flush=True); tot = 0.0
    model.eval()


@torch.no_grad()
def sample(model, prompt_ids, n, temp, max_new):
    """n samples for one prompt (batched; all rows share the prompt length)."""
    x = torch.tensor([prompt_ids] * n)
    done = torch.zeros(n, dtype=torch.bool); outs = [[] for _ in range(n)]
    for _ in range(max_new):
        lg = model(x)[:, -1]
        nxt = lg.argmax(-1) if temp <= 0 else torch.multinomial(torch.softmax(lg / temp, -1), 1)[:, 0]
        for i in range(n):
            if not done[i]:
                t = int(nxt[i])
                if t == EOS: done[i] = True
                else: outs[i].append(t)
        x = torch.cat([x, nxt[:, None]], 1)
        if all(done[i] or (outs[i] and outs[i][-1] in NL_IDS) for i in range(n)):
            break
    return outs


NL_IDS = set()


def decode(ctx, ids):
    return ctx.tok.decode(ids, skip_special_tokens=False)


def heldout(ctx, model, max_new):
    by = {}
    for r in ctx.test:
        o = sample(model, ctx.enc(r["prompt"]), 1, 0.0, max_new)[0]
        by.setdefault(r["task"], []).append(grade(r["prompt"], decode(ctx, o)) == 1.0)
    acc = {k: round(sum(v) / len(v), 4) for k, v in by.items()}
    acc["all"] = round(sum(sum(v) for v in by.values()) / len(ctx.test), 4)
    return acc


@torch.no_grad()
def old_val_loss(ctx, model, n_windows, L=256):
    tot = 0.0
    for b in range(0, n_windows, 8):
        idx = range(b, min(n_windows, b + 8))
        x = torch.stack([torch.from_numpy(ctx.val[i * L:(i + 1) * L + 1].astype("int64")) for i in idx])
        lg = model(x[:, :-1])
        tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)).item() * len(idx)
    return round(tot / n_windows, 4)


def evaluate(ctx, model, stage, rnd, **extra):
    t0 = time.time()
    ctx.log(stage=stage, round=rnd, heldout_acc=heldout(ctx, model, ctx.a.max_new),
            old_val_loss=old_val_loss(ctx, model, ctx.a.val_windows), eval_s=round(time.time() - t0), **extra)


def seq_logprobs(model, x, m):
    lg = model(x[:, :-1]).float()
    lp = torch.log_softmax(lg, -1)
    tok = lp.gather(-1, x[:, 1:, None])[..., 0]
    return tok, lp, m[:, 1:]


def grpo_round(ctx, model, ref, steps, rnd):
    a = ctx.a
    opt = torch.optim.AdamW(model.parameters(), lr=a.rl_lr, weight_decay=0.0)
    rs, kls, fracs = [], [], []
    for step in range(steps):
        prompts = ctx.rng.sample(ctx.train, a.rl_prompts)
        loss_tot = 0.0
        for r in prompts:
            pids = ctx.enc(r["prompt"])
            model.eval()
            outs = sample(model, pids, a.group, a.temp, a.max_new)
            rew = torch.tensor([grade(r["prompt"], decode(ctx, o)) for o in outs])
            rs.extend(rew.tolist())
            adv = (rew - rew.mean()) / (rew.std() + 1e-4) if rew.std() > 0 else torch.zeros_like(rew)
            T = len(pids) + max(1, max(map(len, outs)))
            x = torch.full((a.group, T), EOS, dtype=torch.long); m = torch.zeros(a.group, T)
            for i, o in enumerate(outs):
                s = pids + o; x[i, :len(s)] = torch.tensor(s)
                m[i, len(pids):len(pids) + max(1, len(o))] = 1  # empty output: score the EOS
            model.train()
            tok_lp, _, mm = seq_logprobs(model, x, m)
            with torch.no_grad():
                ref_lp, _, _ = seq_logprobs(ref, x, m)
            d = ref_lp - tok_lp
            kl = (torch.exp(d) - d - 1)  # k3 estimator, >= 0
            ntok = mm.sum(1).clamp(min=1)
            pg = -(adv[:, None] * tok_lp * mm).sum(1) / ntok
            klt = (kl * mm).sum(1) / ntok
            loss = (pg + a.kl_beta * klt).mean() / len(prompts)
            loss.backward(); loss_tot += loss.item()
            kls.append(klt.mean().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        print(f"[rl r{rnd}] step {step+1}/{steps} reward_mean {np.mean(rs[-a.rl_prompts*a.group:]):.3f} "
              f"kl {np.mean(kls[-a.rl_prompts:]):.4f}", flush=True)
    model.eval()
    return round(float(np.mean(rs)), 4), round(float(np.mean(kls)), 5)


def save(ctx, model, cfg, name):
    torch.save({"model": model.state_dict(), "config": {k: getattr(cfg, k) for k in dir(cfg)
                if not k.startswith("_") and not callable(getattr(cfg, k))}}, ctx.out / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--replay", type=int, default=2)
    ap.add_argument("--replay-len", type=int, default=128); ap.add_argument("--replay-weight", type=float, default=0.5)
    ap.add_argument("--sft-steps", type=int, default=400); ap.add_argument("--sft-lr", type=float, default=3e-4)
    ap.add_argument("--ei-rounds", type=int, default=2); ap.add_argument("--ei-prompts", type=int, default=200)
    ap.add_argument("--ei-steps", type=int, default=150); ap.add_argument("--ei-lr", type=float, default=1e-4)
    ap.add_argument("--rl-rounds", type=int, default=3); ap.add_argument("--rl-steps", type=int, default=10)
    ap.add_argument("--rl-prompts", type=int, default=8); ap.add_argument("--rl-lr", type=float, default=2e-5)
    ap.add_argument("--group", type=int, default=4); ap.add_argument("--kl-beta", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--max-new", type=int, default=14)
    ap.add_argument("--val-windows", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    ctx = Ctx(a)
    NL_IDS.update(ctx.nl)
    model, cfg, _ = load_model(a.ckpt)
    ctx.log(stage="config", args=vars(a), train_examples=len(ctx.train), test_examples=len(ctx.test))
    evaluate(ctx, model, "base", 0)

    pool = list(ctx.train)
    sft(ctx, model, pool, a.sft_steps, a.sft_lr, "sft")
    evaluate(ctx, model, "sft", 0); save(ctx, model, cfg, "sft.pt")

    for rnd in range(1, a.ei_rounds + 1):
        prompts = ctx.rng.sample(ctx.train, a.ei_prompts); kept = 0; n = 0
        for r in prompts:
            for o in sample(model, ctx.enc(r["prompt"]), a.group, a.temp, a.max_new):
                n += 1; txt = decode(ctx, o).split("\n")[0]
                if grade(r["prompt"], txt) == 1.0:
                    pool.append({"task": r["task"], "prompt": r["prompt"], "answer": " " + txt.strip(), "src": f"ei{rnd}"})
                    kept += 1
        sft(ctx, model, pool, a.ei_steps, a.ei_lr, f"ei{rnd}")
        evaluate(ctx, model, "expert_iteration", rnd, sampled=n, verified_kept=kept,
                 sample_acc=round(kept / n, 4), pool=len(pool))
    save(ctx, model, cfg, "ei.pt")

    ref = copy.deepcopy(model).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    for rnd in range(1, a.rl_rounds + 1):
        rmean, kl = grpo_round(ctx, model, ref, a.rl_steps, rnd)
        evaluate(ctx, model, "grpo", rnd, reward_mean=rmean, kl_to_ref=kl)
    save(ctx, model, cfg, "rl.pt")


if __name__ == "__main__":
    main()
