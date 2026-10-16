#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Measure params, peak RSS and train tokens/s for a candidate config (AdamW fp32)."""
import json, resource, sys, time, pathlib
import torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pytorch-4b"))
from model import Transformer4B, SmallConfig
h, L, nh, kv, ffn, seq, bs, steps, thr = map(int, sys.argv[1:10])
torch.set_num_threads(thr); torch.manual_seed(0)
cfg = type("P", (SmallConfig,), dict(hidden_dim=h, n_layers=L, n_heads=nh, n_kv_heads=kv, ffn_dim=ffn, max_seq_len=seq))()
m = Transformer4B(cfg); n = sum(p.numel() for p in m.parameters())
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
ts = []
for i in range(steps):
    x = torch.randint(0, 32000, (bs, seq + 1)); t = time.time()
    loss = F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
    opt.zero_grad(); loss.backward(); opt.step(); ts.append(time.time() - t)
st = sorted(ts[2:])[len(ts[2:]) // 2]
print(json.dumps({"hidden": h, "layers": L, "heads": nh, "kv": kv, "ffn": ffn, "seq": seq, "batch": bs, "threads": thr,
                  "params_M": round(n / 1e6, 1), "peak_rss_GB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                  "median_step_s": round(st, 2), "tok_per_s": round(bs * seq / st)}))
