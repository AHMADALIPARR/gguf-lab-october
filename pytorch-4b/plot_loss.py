#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Render train.py's log.csv into docs/train-loss.png.

Usage:
    python3 plot_loss.py --log runs/small-cpu/log.csv [--out docs/train-loss.png]
                         [--unigram 7.1]   # optional baseline line (nats)
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "runs/small-cpu/log.csv"))
    ap.add_argument("--out", default=str(ROOT / "docs/train-loss.png"))
    ap.add_argument("--unigram", type=float, default=None)
    ap.add_argument("--title", default="pytorch-4b small config (34.0M params), CPU run")
    args = ap.parse_args()

    ts, tl, vs, vl = [], [], [], []
    with open(args.log) as f:
        for r in csv.DictReader(f):
            if not r.get("step") or r["step"] == "step":  # skip repeated headers
                continue
            s = int(r["step"])
            ts.append(s); tl.append(float(r["train_loss"]))
            if r.get("val_loss"):
                vs.append(s); vl.append(float(r["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(ts, tl, label="train loss (mean over log interval)", lw=1.5)
    if vs:
        ax.plot(vs, vl, "o-", label="val loss (sampled windows)", lw=1.5, ms=3)
    if args.unigram is not None:
        ax.axhline(args.unigram, ls="--", c="gray", lw=1, label=f"unigram baseline ({args.unigram:.2f})")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title(args.title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out} ({len(ts)} train points, {len(vs)} val points)")


if __name__ == "__main__":
    main()
