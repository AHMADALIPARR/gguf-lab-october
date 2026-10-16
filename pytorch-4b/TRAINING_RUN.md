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

Plot: `docs/train-loss-v1.png` (from `plot_loss.py`).

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

---

# v2: re-corpus + devflow-finance-twin, retrained from scratch

Same `small` config (34,018,176 parameters), same tokenizer, same
hyperparameters except step count and eval. Output is in `runs/small-cpu-v2`
(not committed).

## Data (`scripts/build_corpus_v2.py`)

Text files were extracted from a devflow-finance-twin checkout (master,
d08690f, 1,656 files):

| Outcome | Files |
|---|---|
| Kept as new docs | 1,411 |
| Skipped: exact-hash duplicate (21 of them already in re-corpus, the rest repeated within devflow, including the repeated LICENSE copies) | 31 |
| Skipped: `*.license` REUSE sidecar files (per-file SPDX boilerplate) | 159 |
| Skipped: binary or not valid UTF-8 | 28 |
| Skipped: over 1 MB | 10 |
| Skipped: empty | 14 |
| Skipped: lockfile | 3 |

The split is by document. The v1 val docs are reused unchanged: `val.bin` is
byte-identical to v1 (same md5).

| Split | Docs | Tokens |
|---|---|---|
| train (`tokens.bin`) | 1,958 (618 re-corpus + 1,340 devflow) | 5,506,456 (1,665,029 re-corpus + 3,841,427 devflow) |
| val, old (`val.bin`, the v1 val docs) | 32 | 161,689 |
| val, new (`val-new.bin`, 5% of the devflow docs) | 71 | 214,872 |

## Training

`train.py` gained two options: `--eval-batches 0` evaluates the **full** val
split, and `--max-hours` sets a wall-time cap. v2 evaluated the full old-val
split every 250 steps and saved `best.pt` on that number.

```sh
python pytorch-4b/train.py --config small --data pytorch-4b/data/tokens.bin \
  --val-data pytorch-4b/data/val.bin --out runs/small-cpu-v2 --seq-len 256 \
  --batch-size 16 --total-steps 5400 --warmup-steps 200 --peak-lr 1e-3 \
  --min-lr 1e-4 --log-every 50 --eval-every 250 --eval-batches 0 \
  --ckpt-every 100000 --max-hours 2.85 --workers 0 --threads 8 --epochs 99
```

- The run was planned for 5,400 steps (about 4.0 passes, at 1,344 steps per
  epoch). The 2.85 h cap stopped it at **step 4,718**: 19.3M tokens seen, about
  3.5 passes. The cosine schedule had not reached its floor (LR was about
  1.4e-4 at the stop).
- Wall time: **10,280 s** (2.86 h), as measured by the process clock. This
  includes periods when the shared box was paused or contended. The process
  CPU-time and elapsed counters suggest the pure compute time was lower.

## Loss curve (full old-val split, every 250 steps)

| Step | Epoch | Train (50-step mean) | Val (full) |
|---|---|---|---|
| 250 | 0 | 5.2095 | 5.3292 |
| 1000 | 0 | 3.5541 | 3.8284 |
| 2000 | 1 | 2.8396 | 3.2581 |
| 3000 | 2 | 2.2835 | 2.8288 |
| 4000 | 2 | 2.1220 | 2.5931 |
| 4250 | 3 | 1.8563 | 2.5943 |
| 4500 | 3 | 1.7870 | 2.5560 |
| 4718 (final) | 3 | – | 2.5248 |

Plot: `docs/train-loss.png` (v2, full-split val). The v1 plot is now
`docs/train-loss-v1.png`.

## Full-split evaluation (`eval.py`)

| Checkpoint | Old val loss / ppl | New val loss / ppl |
|---|---|---|
| v1 `best.pt` (step 2000) | 4.7251 / 112.74 | – |
| v2 `best.pt` (step 4500) | 2.5560 / 12.88 | 2.7682 / 15.93 |
| v2 `final.pt` (step 4718) | **2.5248** | **2.7637 / 15.86** |
| Unigram baseline (v2 train counts) | 7.6256 / 2050.01 | 7.3293 / 1524.25 |

- **Old val:** on the same 32 docs, full-split val loss fell from 4.7251 (v1)
  to 2.5248 (v2 final), a drop of 2.20 nats/token.
- **Overfitting:** unlike v1, val loss was still falling when the run stopped.
  The train/val gap at step 4500 was about 0.77 nats, against about 3 nats in
  v1. The run did not overfit within its 3.5 passes. More steps would probably
  lower val loss further, but this run did not measure that.
- **Caveats:**
  - Part of the improvement on old val may come from the overlap between the
    corpora: 21 devflow files were already in re-corpus, and related devflow
    text now appears in train. Exact duplicates were removed, so the old val
    docs themselves are not in train.
  - The unigram baseline changed, from 7.7180 to 7.6256, because the train
    counts changed.
- **Checkpoint choice:** `final.pt` scores lower than `best.pt` on both splits.
  `best.pt` was only updated at 250-step evals, and the final steps continued
  to improve. `final.pt` is the lowest-loss v2 checkpoint, and it is the one
  exported.

## GGUF export

```sh
python pytorch-4b/export_gguf.py runs/small-cpu-v2/final.pt runs/small-cpu-v2/small-v2-final.gguf --config small
```

- 137,155,936 bytes, F32, rope theta 10,000 read from the checkpoint. Not
  committed.
- llama.cpp `llama-simple` loads and generates from the file. For the prompt
  below, its greedy output is identical to PyTorch greedy decoding.

Prompt `# Reverse engineering` (llama-simple, greedy, 40 tokens):

```
# Reverse engineering
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--get-parse", "module")
    parser.add_
```

Output is syntactically plausible Python and Markdown, but it does not answer
the prompts. On question prompts, greedy decoding still repeats itself (see
`eval.py` samples).

# v3: v2 code/RE data + TinyStories prose (small, CPU)

## Data (`scripts/build_corpus_v3.py`, output in `pytorch-4b/data/v3/`)

- Train stream `v3/tokens.bin` = 11,006,606 tokens:
  - v2 `tokens.bin`, copied byte-for-byte: 5,506,456 tokens (re-corpus 1,665,029 + devflow 3,841,427);
  - TinyStories (`roneneldan/TinyStories`, `data/train-00000-of-00004` parquet, file order, exact-dedup): 19,692 stories, 5,500,150 tokens.
- `v3/val-stories.bin`: 1,000 stories from the official TinyStories validation split (seed-2 shuffle, none duplicated in train), 268,246 tokens.
- `val.bin` and `val-new.bin` were not changed. The tokenizer is the same (vocab 32000, EOS 2).

## Training

The config and hyperparameters match v2 (`small`, seq 256, batch 16, peak LR 1e-3, min LR 1e-4, warmup 200, total-steps 4718, 8 threads, seed 0).
The run validated on `val.bin` every 250 steps. The 2.85 h cap stopped it at **step 4,648** (epoch index 1, about 1.7 passes) after a wall time of 10,760 s.

## Full-split evaluation (`eval.py`)

| Checkpoint | val.bin loss / ppl | val-new.bin loss / ppl | val-stories.bin loss / ppl |
|---|---|---|---|
| v2 `final.pt` (step 4718) | 2.5248 / 12.5 | 2.7637 / 15.86 | 8.5912 / 5383.82 |
| v3 `best.pt` (step 4500) | 3.2248 / 25.15 | 3.1297 / 22.87 | 2.0385 / 7.68 |
| v3 `final.pt` (step 4648) | 3.2196 / 25.02 | 3.1125 / 22.48 | **2.0333 / 7.64** |

- On code/RE, v3 is worse than v2: about +0.69 nats on val.bin and +0.35 on val-new.bin. It saw about half as many code tokens per pass and made about 1.7 passes over the code data, against about 3.5 for v2.
- On stories, v3 is strong, and v2 is near unigram level.

## GGUF export

`runs/small-cpu-v3/small-v3-final.gguf` (from `final.pt`): 137,155,936 bytes, F32.

## Samples (`scripts/sample_small.py`, final.pt, 60 tokens)

Greedy, prompt "Once upon a time, there was a little girl named Lily.":

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big tree with a big, red apple. She wanted to eat it, but she was too scared to climb it.
```

Greedy, prompt `def read_config(path):`:

```
def read_config(path):
    return {
        "path": "src/a68/main_stack.a68",
        "src/a68/main.a68",
        "src/a68/a68/a68/a68/a68/a68/a68/a68/
```

Story prompts give grammatical, mostly coherent short prose, with occasional logic slips. Code prompts are still incoherent and fall into repetition loops.
