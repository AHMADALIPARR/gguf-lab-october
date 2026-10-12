<!--
  Copyright (C) 2026 Ahmad Ali Parr
  SPDX-License-Identifier: AGPL-3.0-only
-->

# GGUF Lab — October 2026 Drops

Pruning + compression lab for the October 2026 GGUF releases.
Staged here, executed on real hardware (this VM has no GPU and cannot
hold these models — pull and run happen on the NUC fleet / main rig).

## The six drops (all verified live on HuggingFace)

| # | Model | HF repo (GGUF) | Params | Pick for |
|---|---|---|---|---|
| 1 | Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF` | 27B dense | Coding / agents |
| 2 | Qwen3.8 Flash Next | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | 177B MoE | Low-bit MoE research (offload only) |
| 3 | Ternary Bonsai 2 27B | `prism-ml/Ternary-Bonsai-2-27B-gguf` | 27B ternary | Weight-compression research |
| 4 | Mellum 2.1 Thinking | `JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF` | 12B / 2.5B active MoE | Coding, fits small VRAM |
| 5 | EmbeddingGemma 2 | `unsloth/embeddinggemma-2-GGUF` | 300M | Retrieval / agent memory |
| 6 | Xing 4.0 29B-A4B | `XingChen-AGI/Xing4.0-29B-A4B-GGUF` | 29B / 4B active MoE | Perf-per-token |

Key files (verified in repo listings):
- Qwen3.8-27B: `Qwen3.8-27B-UD-Q4_K_M.gguf`, `Qwen3.8-27B-UD-Q4_K_S.gguf`, `Qwen3.8-27B-Q8_0.gguf`
- Mellum 2.1: `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf` (official JetBrains)
- EmbeddingGemma 2: via `unsloth/embeddinggemma-2-GGUF` or `ggml-org/embeddinggemma-2-GGUF`

## Hardware mapping

| Hardware | Target models |
|---|---|
| NUC (11B cap) | EmbeddingGemma 2, Mellum 2.1 Q4 (offload) |
| 2nd NUC (arrives tomorrow) | Pipeline-split candidate |
| Main rig (big VRAM) | Qwen3.8-27B, Xing 4.0, Ternary Bonsai 2, Glimmer 30B |
| 177B Flash Next | Offload-only everywhere; research target, not daily driver |

## Pipeline

```
pull.sh → pull GGUFs to local store
   │
prune/plan.md → structured pruning per model (layer drop, width, heads)
   │
compress/plan.md → quantization targets (Q4_K_M baseline, ternary study)
   │
GGUF out → Ollama / llama.cpp
```

## Contents

- `MANIFEST.md` — full model inventory with arch notes
- `pull.sh` — download commands (run on the target machine)
- `prune/plan.md` — pruning plan per model
- `compress/plan.md` — compression/quantization plan

## License

AGPL-3.0-only. Tools referenced (sovereign-mimo-4b, tensor-roll) keep
their own licenses.
