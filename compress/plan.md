<!--
  Copyright (C) 2026 Ahmad Ali Parr
  SPDX-License-Identifier: AGPL-3.0-only
-->

# Compression plan

## Streaming expert-selection exception

The implemented Mellum E8 experiment selects complete existing quantized expert
blocks byte-for-byte. It does not change the retained blocks' quantization or
introduce another rounding pass. Its main risk is loss of expert coverage and
changed router normalization, which still requires quality evaluation. This
exception does not permit arbitrary quantized width/head projection, and the
full-precision workflow below remains necessary for those transformations.

Quantization targets per model. Baseline: Q4_K_M everywhere unless noted.
Tool: llama.cpp `llama-quantize` on the main rig.

## Targets

| Model | Pull as | Further compression |
|---|---|---|
| EmbeddingGemma 2 | Q4_K_M | None needed (300M) |
| Mellum 2.1 | Q4_K_M (official) | IQ4_NL / Q3_K_M for NUC after prune |
| Qwen3.8-27B | UD-Q4_K_M | IQ quants post-prune |
| Xing 4.0 | IQ4_NL (official) | As-is; MoE already sparse |
| Ternary Bonsai 2 | PTQ1_0 | None — study, don't requantize |
| Flash Next 177B | — | Offload; no local quant run |

## Notes

- "UD" prefix on Unsloth files = their updated dynamic quantization.
  Prefer UD-Q4_K_M over plain Q4_K_M where both exist.
- Ternary models (Bonsai): the PTQ1_0/PQ2_0 files ARE the compression.
  Running them through another quantizer destroys the ternary structure.
- Embedding models: keep Q4_K_M or higher — retrieval quality is
  sensitive to embedding degradation.
- After any prune: re-quantize from the pruned BF16/F16, never
  quantize-then-prune (pruning quantized weights compounds error).
