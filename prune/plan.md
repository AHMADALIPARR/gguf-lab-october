<!--
  Copyright (C) 2026 Ahmad Ali Parr
  SPDX-License-Identifier: AGPL-3.0-only
-->

# Pruning plan

## Implemented low-memory path

`scripts/prune_mellum.py` implements streaming selection of complete expert banks
and router rows from the pinned Q4 GGUF. It retains 8 of 64 experts in each of
28 layers using uncalibrated router-row L2 norms. It preserves retained weight
bytes and all tokenizer metadata, checks a 1.5 GiB file budget, and uses no GPU.
This is an experimental candidate, not the activation-calibrated PyTorch recipe
below. See `NUC-2GB.md` and `reports/2gb-plan.json` for supported and blocked models.

Structured pruning per model, adapted from the sovereign-mimo-4b recipe
(layer drop by attention variance → width/head projection → MLP shrink →
GGUF). Scripts live in `SNAPKITTYWEST/sovereign-mimo-4b` (`prune/`):
retarget per architecture below. Nothing here runs on the staging VM —
execute on the main rig.

## Order (fast iteration first)

### 1. Mellum 2.1 (12B/2.5B MoE) — first real target
- Small enough to iterate quickly. MoE: prune at expert granularity —
  drop lowest-utility experts first (router-weight analysis), then
  structured width pruning on shared blocks.
- Watch: MoE routing collapse. Validate tool-use/coding benchmarks
  after each cut, not just perplexity.

### 2. Qwen3.8-27B — main workhorse
- Dense 27B. mimo-4b recipe directly applicable: layer-drop by
  attention variance, hidden-width projection, head prune, MLP shrink.
- Target: fit the 11B NUC (roughly 40% of params) or a comfortable
  main-rig footprint.

### 3. Xing 4.0 29B-A4B — MoE
- Same expert-first approach as Mellum. 4B active means the MoE
  sparsity is already doing heavy lifting; prune experts, not width.

### 4. Ternary Bonsai 2 27B — do NOT conventionally prune
- Already ternary (≈1.58 bits/weight). Standard pruning/quantization
  is the wrong tool — study the ternary representation instead
  (sensitivity analysis per layer, not requantization).

### 5. Flash Next 177B — offload-only, prune last
- No local pruning run is feasible. Research target only.

## Agentic-localization gate (Glimmer + coding models)

Before pruning any agentic model (Glimmer 30B, Qwen3.8-27B for agent
workflows): run the ablation harness FIRST (see the Glimmer targeting
brief: 13 global layers first, Q-head-output granularity, never KV
heads). Prune around the localized circuits, not through them.

## Tooling

- `sovereign-mimo-4b/prune/` — structured pruning scripts (PyTorch)
- `sovereign-gemini-gguf` — GGUF topology mapping without loading weights
- `tensor-roll` — 30B→8B distillation design (toy-scale validated only)
- `hologram-re` B_elision — staged execution under memory budgets
