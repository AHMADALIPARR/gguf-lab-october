<!--
  Copyright (C) 2026 Ahmad Ali Parr
  SPDX-License-Identifier: AGPL-3.0-only
-->

# CPU training run: `small` config of the pytorch-4b architecture

A scaled-down copy of the same architecture as the 3.82B default (GQA,
SwiGLU, RMSNorm, RoPE, 32k vocab), trained on CPU on the repository's
RE corpus. Every number below comes from this run's logs (`log.csv`,
`report.json`) or from `eval.py`. Nothing is extrapolated.

## Environment

- CPU only: 8 cores, ~16 GB RAM (shared with other jobs), no GPU.
- torch 2.14.1+cpu, tokenizers 0.23.3, gguf 0.19.0, in a venv at `/workspace/venv-cpu`.
- Training process RSS peaked at about 3.3 GB.
- Other jobs (a llama.cpp build and evaluation scripts) shared the CPU during
  part of the run, so step times varied between about 1 s and 4 s.

## Model (`--config small`, `model.py:SmallConfig`)

| Param | Value |
|---|---|
| Parameters | **34,018,176** (untied embedding and output head: 2 x 12,288,000) |
| Layers | 6 |
| Hidden dim | 384 |
| Heads | 6 Q / 2 KV (GQA 3:1), head dim 64 |
| FFN | 1024 (SwiGLU) |
| Vocab | 32,000 |
| Context | 256 |
| RoPE theta | 10,000 (stored in every checkpoint's `config`) |

The 4B default (`Config`, `--config 4b`) is unchanged.

Code changes made for this run:
- `model.py`: added `SmallConfig`, `CONFIGS`, and `config_to_dict`. Attention now
  uses `F.scaled_dot_product_attention`, which is **causal by default**. The
  previous code applied no mask when `mask=None`, and `train.py` never passed
  one, so the original loop would have trained a non-causal model.
- `train.py`: added `--config`, `--val-data`, `--eval-every`, `--eval-batches`,
  `--threads`, and `--seed`. It now writes a CSV log, saves `best.pt` on the
  sampled val loss, and does a full-split val eval at the end. Checkpoints
  embed the config dict.

## Data

`scripts/tokenize_split.py` splits `data/re-corpus.jsonl` by document
(seeded shuffle, 5% of documents to val) and appends EOS (`</s>`, id 2) after
each document. See `data/split_stats.json` for the counts.

| Split | Docs | Tokens |
|---|---|---|
| train (`tokens.bin`) | 618 | 1,665,029 |
| val (`val.bin`) | 32 | 161,689 |

The val split is 5% of documents but 8.9% of tokens. The 70.6M-token
`tokens.bin` mentioned in README.md was not available. This run uses only the
650-document corpus.

## Training

```sh
python pytorch-4b/train.py --config small \
  --data pytorch-4b/data/tokens.bin --val-data pytorch-4b/data/val.bin \
  --out runs/small-cpu --seq-len 256 --batch-size 16 --total-steps 2400 \
  --warmup-steps 100 --peak-lr 1e-3 --min-lr 1e-4 --log-every 50 \
  --eval-every 100 --eval-batches 20 --ckpt-every 600 --workers 0 \
  --threads 8 --epochs 99
```

- 2,400 steps x 16 x 256 = 9,830,400 training tokens. That is about 5.9 passes
  over the 1,665,029 train tokens. Each epoch is 406 steps.
- AdamW (betas 0.9/0.95, weight decay 0.1), cosine LR from 1e-3 to 1e-4,
  100 warmup steps, grad clip 1.0.
- Wall time: **4,682 s** (78 min), including evals and checkpoint writes.

## Loss curve

Train is the mean over the previous 50 steps. "Val (sampled)" is the
in-training eval, which scores only the first 320 windows (about half of the
val split). It is lower than the full-split numbers and should not be compared
with them.

| Step | Epoch | Train | Val (sampled) |
|---|---|---|---|
| 100 | 0 | 6.6339 | 6.1893 |
| 400 | 0 | 4.3624 | 4.5802 |
| 800 | 1 | 3.5692 | 4.0574 |
| 1200 | 2 | 2.9800 | 3.8120 |
| 1600 | 3 | 2.4544 | 3.7049 |
| 2000 | 4 | 2.0727 | 3.7010 (min) |
| 2400 | 5 | 1.7107 | 3.7609 |

Plot: `docs/train-loss.png` (from `plot_loss.py`).

## Full-split evaluation (`eval.py`, 161,536 val targets)

| Checkpoint | Step | Val loss | Val ppl |
|---|---|---|---|
| Unigram baseline (train counts, add-one) | – | 7.7180 | 2248.43 |
| `best.pt` | 2000 | **4.7251** | **112.74** |
| `final.pt` | 2400 | 4.8019 | 121.74 |

**The model overfits.** Train loss falls from about 2.0 at step 2000 to 1.71
at step 2400. Over the same steps, full-split val loss rises from 4.7251 to
4.8019. With 34M parameters and 1.7M training tokens, the model is memorizing
the corpus. `best.pt` (step 2000) has the lowest full-split val loss, so it is
the checkpoint that gets exported. See `EVAL_NOTES.md` for per-checkpoint
detail.

## GGUF export

```sh
python pytorch-4b/export_gguf.py runs/small-cpu/best.pt runs/small-cpu/small-best.gguf --config small
```

- The export reads its config from the checkpoint (`rope_theta` 10000, ctx 256)
  and maps tensors as llama arch.
- The output file is 137,155,936 bytes (F32). It is not committed.
- The file loads and generates with llama.cpp `llama-simple` (built from source
  at commit 4413a19). With greedy decoding, llama.cpp produced the same
  continuation as PyTorch greedy decoding (`scripts/sample_small.py`) for the
  prompt below.

## Sample generations (`best.pt` via llama.cpp, greedy, 48 tokens)

Prompt `# Reverse engineering`:

```
# Reverse engineering,
# Its covenant endureth through the dark.
#
# Ignorantia juris non excusat.
# ========================================================================

"""Quantum layer: zero-copy tensor."""

from __future__ import annotations

from dataclasses import dataclass,
```

Prompt `Explain stack canaries and how they prevent buffer overflows:`:

```
    - name = "No"
    - name = "not"
    - name = "not"
    - name = "not"
    ...
```

The model reproduces the surface form of the corpus (file headers, Python,
Markdown). It does not answer questions, and greedy decoding falls into
repetition. This is consistent with the model size and the amount of data.
