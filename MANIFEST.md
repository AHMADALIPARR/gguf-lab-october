<!--
  Copyright (C) 2026 Ahmad Ali Parr
  SPDX-License-Identifier: AGPL-3.0-only
-->

# MANIFEST — October 2026 GGUF Drops

All entries verified live on HuggingFace 2026-10-11. Sizes are
estimates from architecture + quantization; exact bytes confirmed at
pull time.

## 1. Qwen3.8-27B — first pick for coding

- GGUF: `unsloth/Qwen3.8-27B-GGUF`
- Base: `Qwen/Qwen3.8-27B` (check for the official repo)
- 27B dense. Q4_K_M ≈ 16–18 GB. Q8_0 ≈ 29 GB.
- Files: `Qwen3.8-27B-UD-Q4_K_M.gguf` (start here),
  `Qwen3.8-27B-UD-Q4_K_S.gguf`, `Qwen3.8-27B-Q8_0.gguf`
- Target: main rig. Prune candidate: layer drop + width (mimo-4b recipe).

## 2. Qwen3.8 Flash Next 177B — MoE research

- GGUF: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
- Base: `Qwen/Qwen3.8-Flash-Next`
- 177B MoE. Far beyond 10 GB VRAM — offload-only on all local hardware.
- Use: low-bit inference + MoE routing experiments, not a daily driver.

## 3. Ternary Bonsai 2 27B — compression research

- GGUF: `prism-ml/Ternary-Bonsai-2-27B-gguf`
- 27B-class ternary (≈1.58 bits/weight). Already extreme compression —
  pruning here means studying the ternary representation, not re-quantizing.
- Alternates: `CaryPalmer/Ternary-Bonsai-2-27B-262k-GGUF` (262k context).

## 4. Mellum 2.1 Thinking — small-VRAM coding

- GGUF: `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF` (official JetBrains)
- 12B total / 2.5B active MoE. Q4_K_M ≈ 7–9 GB.
- Files: `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf` (start here)
- Target: NUC with offload, or main rig native. Strong prune candidate
  (small enough to iterate fast).

## 5. EmbeddingGemma 2 — agent memory

- GGUF: `unsloth/embeddinggemma-2-GGUF` (also `ggml-org/embeddinggemma-2-GGUF`)
- 300M. Runs anywhere, including the 11B NUC with room to spare.
- Use: retrieval, semantic indexing, local agent memory.

## 6. Xing 4.0 29B-A4B — perf-per-token MoE

- GGUF: `XingChen-AGI/Xing4.0-29B-A4B-GGUF` (official)
- 29B total / 4B active. MoE sparsity play.
- Target: main rig.

## Pull priority (pruning/compression order)

1. EmbeddingGemma 2 — trivial, validates the pipeline
2. Mellum 2.1 Q4_K_M — first real prune target (fast iteration)
3. Qwen3.8-27B Q4_K_M — main coding workhorse
4. Xing 4.0 — MoE experiment
5. Ternary Bonsai 2 — compression research
6. Flash Next 177B — offload-only, last
