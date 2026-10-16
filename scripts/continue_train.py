#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Continue-train a checkpoint (e.g. the four-intent model) on a token
stream, with a fraction of each batch drawn from the old train stream to
limit forgetting. Harness code. Logs train loss, held-out loss on the new
corpus (last --val-frac of tokens, never trained on) and old-val loss.
Saves ckpt-<step>.pt every --ckpt-every steps and final.pt.
"""
import argparse, json, math, pathlib, sys, time
import numpy as np, torch, torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parents[1] / "pytorch-4b"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "dss"))
from eval import load_model  # noqa: E402


def windows(arr, n, L, rng):
    i = rng.integers(0, len(arr) - L - 1, n)
    x = np.stack([arr[j:j + L + 1] for j in i]).astype(np.int64)
    return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


@torch.no_grad()
def loss_on(model, arr, n, L, seed):
    rng = np.random.default_rng(seed); model.eval(); tot = 0.0
    for _ in range(n // 8):
        x, y = windows(arr, 8, L, rng)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train(); return tot / (n // 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--old-data", required=True)
    ap.add_argument("--old-val", required=True)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5); ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--old-frac", type=float, default=0.25); ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--init", default=None, help="fresh init: hidden,layers,heads,kv,ffn (ignores --ckpt)")
    ap.add_argument("--accum", type=int, default=1); ap.add_argument("--val-windows", type=int, default=128)
    ap.add_argument("--max-hours", type=float, default=0); ap.add_argument("--val-data", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.init:
        from model import Transformer4B, SmallConfig
        h, L, nh, kv, ffn = map(int, a.init.split(","))
        cfg = {k: getattr(SmallConfig, k) for k in dir(SmallConfig) if not k.startswith("_")}
        cfg.update(hidden_dim=h, n_layers=L, n_heads=nh, n_kv_heads=kv, ffn_dim=ffn, max_seq_len=a.seq)
        model = Transformer4B(type("C", (), cfg)())
        print(f"fresh init {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
    else:
        model, cfg, _ = load_model(a.ckpt)
    model.train()
    new = np.fromfile(a.data, dtype=np.uint16); cut = int(len(new) * (1 - a.val_frac))
    tr, va = new[:cut], new[cut:]
    if a.val_data: tr, va = new, np.fromfile(a.val_data, dtype=np.uint16)
    old = np.fromfile(a.old_data, dtype=np.uint16); oval = np.fromfile(a.old_val, dtype=np.uint16)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    log = open(out / "log.jsonl", "a")
    def ev(step):
        r = {"step": step, "new_val": round(loss_on(model, va, a.val_windows, a.seq, 1), 4),
             "old_val": round(loss_on(model, oval, a.val_windows, a.seq, 2), 4), "time": time.strftime("%H:%M:%S")}
        log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
    log.write(json.dumps({"config": vars(a), "new_tokens_train": int(len(tr)), "new_tokens_val": int(len(va)),
                          "old_tokens": int(len(old))}) + "\n")
    ev(0); t0 = time.time(); run = 0.0
    n_old = int(round(a.batch * a.old_frac))
    for step in range(1, a.steps + 1):
        lr = a.lr * step / a.warmup if step <= a.warmup else \
            a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup)))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad()
        for _ in range(a.accum):
            x1, y1 = windows(tr, a.batch - n_old, a.seq, rng); x2, y2 = windows(old, n_old, a.seq, rng)
            x, y = torch.cat([x1, x2]), torch.cat([y1, y2])
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten()) / a.accum
            loss.backward(); run += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:
            ev(step); a.steps = step
        if step % 25 == 0:
            print(f"[ct] step {step}/{a.steps} loss {run / 25:.4f} lr {lr:.2e} {time.time() - t0:.0f}s", flush=True); run = 0.0
        if step % a.eval_every == 0 or step == a.steps: ev(step)
        if step % a.ckpt_every == 0 or step >= a.steps:
            name = "final.pt" if step >= a.steps else f"ckpt-{step}.pt"
            torch.save({"model": model.state_dict(), "config": cfg if isinstance(cfg, dict) else
                        {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}},
                       out / name)
        if step >= a.steps: break


if __name__ == "__main__":
    main()
