---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: feature-extraction
base_model: unsloth/embeddinggemma-2-GGUF
tags:
- gguf
- embeddings
- gemma-embedding2
---

# EmbeddingGemma 2 Q4 — unchanged small artifact

This repository mirrors the unchanged `embeddinggemma-2-UD-Q4_K_XL.gguf` from
[unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF),
revision `031f0d4b35536f69ab3509d4893c923264fcf253`.
No model weights were pruned or modified: its 175,673,856-byte size already meets
the small-model budget. The upstream SHA256 was verified.

SHA256: `ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493`.

## CPU smoke test

llama.cpp b11577 CPU-only returned three finite 768-dimensional text embeddings.
A related text pair had cosine similarity 0.825; an unrelated pair scored 0.559.
This small smoke test is not a retrieval benchmark. Peak observed process working
set was 278,257,664 bytes. No GPU layers were used, and the server was stopped
after testing. b11429 did not recognize the `gemma-embedding2` architecture.

Use a compatible recent runtime, one thread, one slot, 256-token context and
128-token batch/microbatch. This artifact covers text embeddings; a multimodal
projector is not included.

## Attribution and license

All pretrained-model credit remains with the upstream authors. This is a
verified mirror of their artifact, not a newly trained model.
Apache-2.0 is preserved as declared in the model's GGUF metadata. Lab scripts
are separately AGPL-3.0-only. See `LICENSE`, `NOTICE` and `smoke-test.json`.

[GGUF Lab code and reports](https://github.com/AHMADALIPARR/gguf-lab-october/tree/small-vram-pruning).
