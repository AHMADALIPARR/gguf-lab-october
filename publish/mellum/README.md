---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model: JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF
tags:
- gguf
- mellum
- pruned
- experimental
---

# Mellum 2.1 E8 — experimental GGUF

This is an **uncalibrated expert-pruning experiment**, not a trained smaller
model or a production-ready coding assistant. Coding/tool-use quality, inference
success and runtime VRAM usage have not been measured.

## Artifact

`mellum-E8-experimental.gguf`: 1,582,936,736 bytes, 80.4% smaller than the
8,071,295,264-byte source. All 28 layers and 339 tensor names are retained.
The expert pool changes from 64 to 8 per layer; top-8 routing stays unchanged.
Experts are selected by squared L2 norms of their router rows, not activation
statistics. Expert-bank slices and router rows are copied byte-for-byte without
requantization. Tokenizer and upstream model-license metadata are preserved.
Expert coverage and routing distributions change, so quality may degrade severely.

SHA256: `4eed7d5e0915d34fd789fc3ba86e431b8f38196dc486253c84e4c71f7409092e`.

## Provenance and validation

Source: [JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF),
revision `20b439426f1e997ba575eaccaf4856ac0f76f5d5`, Q4_K_M artifact.
Modified by Ahmad Ali Parr with Codex, 12 October 2026.

Independent output SHA256, tensor names, shapes, types and byte boundaries were
checked. Synthetic tests verify byte-preserving slicing and router correspondence.
The complete source file was not downloaded/hashed: retained ranges were fetched
from the pinned revision. Its expected upstream hash is recorded as unverified
in `provenance.json`. Structural validation is not inference/quality validation.

## Resource limits

The pruner used zero GPU, 4 MiB transfer chunks and approximately 40.5 MiB peak
working set. Those are processing costs, not model inference costs.
The 1.5 GiB weight allowance leaves planning space within a 2 GiB device, but
**does not prove the model fits within 2 GiB runtime VRAM**.

Full-model inference was skipped because only 812 MB physical RAM was free,
below the 2 GiB guard. For later CPU testing, use one thread, zero GPU layers,
512-token context, 32-token batch/microbatch and one model at a time. Reserve at
least 2 GiB free system RAM first. Do not use this unvalidated candidate for
production tasks.

## Reproduction and license

[GGUF Lab code and reports](https://github.com/AHMADALIPARR/gguf-lab-october/tree/small-vram-pruning).
Run `scripts/audit_models.py`, then `scripts/prune_mellum.py --experts 8`.
The model retains Apache-2.0; the lab scripts retain AGPL-3.0-only.
See `LICENSE`, `NOTICE`, `provenance.json` and `validation.json`.
