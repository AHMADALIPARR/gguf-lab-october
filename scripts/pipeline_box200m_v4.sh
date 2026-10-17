#!/bin/bash
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# Stage pipeline after continue_train (harness). Each step skips if its output exists (resumable).
set -u
R=/workspace/gguf-lab-october; PY=/workspace/venv-cpu/bin/python; V=$R/runs/box-200m-v4; I=$R/runs/box-200m-v4-intent
N="nice -n 19"; note(){ echo "- $(date '+%F %H:%M %Z') $*" >> $V/PROGRESS.md; }
while pgrep -f scripts/continue_train.py >/dev/null; do sleep 60; done
[ -f $V/final.pt ] || { note "final.pt missing after continue_train exit; stopping"; exit 1; }
note "continue_train done (final.pt)"
cd $R/pytorch-4b/dss
if [ ! -f $I/ei.pt ]; then
  mkdir -p $I; note "intent SFT/EI started"
  $N $PY agent.py train --mode intent --ckpt $V/final.pt --out $I --train-file data/train_aug.jsonl \
    --extra-train data/phi_train.jsonl,data/ruby_train.jsonl,data/ruby_train_targeted.jsonl \
    --test-file data/test_ruby_mixed_balanced.jsonl --sft-steps 600 --ei-rounds 1 --ei-prompts 200 --ei-steps 150 \
    --rl-rounds 0 --threads 8 --max-hours 5 > $I/train.log 2>&1 || { note "agent train failed rc=$?"; exit 1; }
  note "intent SFT/EI done"
fi
[ -s $V/ei-old-newharness.json ] || { $N $PY agent.py eval $R/runs/box-200m-intent/ei.pt --mode intent --threads 8 --test-file data/test_ruby_mixed_balanced.jsonl > $V/ei-old-newharness.json 2>$V/ei-old-newharness.err; note "old ei.pt heldout (new harness) done"; }
[ -s $V/ei-new-legacyharness.json ] || { DSS_FIXED_FINAL=1 $N $PY agent.py eval $I/ei.pt --mode intent --threads 8 --test-file data/test_ruby_mixed_balanced.jsonl > $V/ei-new-legacyharness.json 2>$V/ei-new-legacyharness.err; note "new ei.pt heldout (legacy 24-tok harness) done"; }
G=$I/box-200m-v4-intent-ei.gguf
[ -f $G ] || { $N $PY $R/pytorch-4b/export_gguf.py $I/ei.pt $G > $I/export.log 2>&1 && sha256sum $G > $G.sha256; note "gguf exported"; }
mkdir -p $I/live
[ -s $I/live/bal.json ] || { $N $PY host_eval.py intent $G data/test_ruby_mixed_balanced.jsonl $I/live/bal > $I/live/bal.out 2>&1; note "live balanced done"; }
[ -s $I/live/ood.json ] || { $N $PY host_eval.py intent $G data/test_ruby_mixed_ood.jsonl $I/live/ood > $I/live/ood.out 2>&1; note "live ood done"; }
note "pipeline complete"
