<!--
  Copyright (C) 2026 Ahmad Ali Parr
  SPDX-License-Identifier: AGPL-3.0-only
-->

# pytorch-4b — 4B dense transformer

Recreated from the unified tensor manifest pattern (`manifests/unified.jsonl`,
built via hologram-re's `strip` approach). Architecture follows the observed
patterns: GQA attention, SwiGLU FFN, RMSNorm, RoPE.

## Architecture

| Param | Value |
|---|---|
| Parameters | **3.82B** |
| Layers | 36 |
| Hidden dim | 3072 |
| Attention heads | 24 (Q), 8 (KV) — GQA 3:1 |
| FFN dim | 8192 (SwiGLU) |
| Vocab | 32000 |
| Context | 4096 |
| RoPE theta | 500000 |

Derived from: mellum (28 blocks, GQA), qwen27 (SwiGLU, RMSNorm).

## Files

- `model.py` — Transformer4B definition (RMSNorm, GQA, SwiGLU, RoPE)
- `train.py` — training loop (AdamW, cosine schedule, checkpointing)

## Training

Requires GPU with >=40GB VRAM. This VM cannot train (OOM on init).

```sh
python3 train.py --data /path/to/tokens.bin --out checkpoints/ \
    --seq-len 2048 --batch-size 4 --total-steps 100000
```

Dataset: flat binary file of uint16 token ids (`TokenDataset`).
Replace with your own data pipeline as needed.

## Honest status

- Architecture defined and param-counted: **3.82B**
- Forward pass shape-checked on paper (cannot run here — OOM)
- Training loop written, not executed (no GPU, no training data)
- Not a trained model — this is the architecture + loop, weights are random
