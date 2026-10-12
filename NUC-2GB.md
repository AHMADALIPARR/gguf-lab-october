<!-- Copyright (C) 2026 Ahmad Ali Parr; SPDX-License-Identifier: AGPL-3.0-only -->
# NUC with 2 GiB VRAM

This machine reports Radeon 540X, 2 GiB dedicated VRAM and approximately 8 GiB
system RAM. During processing, free system RAM was about 1 GiB. The processing
pipeline uses no GPU, one sequential downloader and 4 MiB weight-copy chunks.
Mellum pruning was observed at approximately 40.5 MiB peak working set. These are
processing measurements, not runtime inference memory measurements.

## Artifacts and method

`models/embeddinggemma-2-UD-Q4_K_XL.gguf` is the unchanged 175,673,856-byte upstream
embedding model, SHA256-verified. No pruning is necessary to meet this budget.
A CPU-only b11577 smoke test returned three finite 768-dimensional embeddings;
the related pair scored 0.825 cosine similarity, versus 0.559 for an unrelated
pair. This is a smoke test, not retrieval-quality evaluation. The temporary
server was stopped. b11429 does not recognize `gemma-embedding2`.

`models/mellum-E8-experimental.gguf` is an experimental expert-pruned candidate.
It keeps all 28 layers and the existing top-8 routing count, reducing the expert
pool from 64 to 8 per layer. Selected experts have the largest router-row squared
L2 norms. This is an uncalibrated parameter heuristic, not measured expert utility.
The selected expert banks and router rows are copied byte-for-byte: no weights
are dequantized or requantized. Existing model license and tokenizer metadata are
preserved. Model quality may degrade severely after this removal.

The completed, independently SHA256-checked output is 1,582,936,736 bytes, within the 1.5 GiB weight allowance.
The remaining 0.5 GiB is only a planning allowance for context/workspace; GPU
offload has not been measured. Model size alone is not a VRAM guarantee.
Coding, tool-use and perplexity gates have not been evaluated. Keep this artifact
out of production until those comparisons pass.

## Reproduce

```text
python scripts/audit_models.py
python scripts/plan_2gb.py
python -m unittest discover -s tests -v
python scripts/download_small.py
python scripts/prune_mellum.py --experts 8
```

Downloads and pruning refuse to overwrite existing artifacts. The source model
revisions and upstream hashes are recorded in `reports/model-audit.json`.
The pruner validates exact HTTP ranges, tensor dimensions, quantization block
sizes, expert/router correspondence, output budget and metadata. It writes a
`.part` file first and only renames after structural validation. Output provenance
includes the selected expert IDs and SHA256. The full 8.1 GB Mellum source is not
downloaded or hashed; its upstream full-file hash is recorded as unverified, while
retained byte ranges are fetched from its pinned immutable revision.

CPU-only inference defaults are in `profiles/2gb-cpu.json`. Load only one model
at a time, keep context small, and retain GPU layers at zero while gaming.
Mellum inference needs more free RAM than the pruning pipeline; allow at least
2 GiB free RAM before testing that candidate.

## Other models

The audit covers all six repositories without loading model weights.
`reports/2gb-plan.json` records architecture and tensor byte totals.

| Model | Why this pipeline does not emit a 2 GiB candidate |
| --- | --- |
| Qwen 27B Q4 | Non-layer embedding/output tensors alone occupy 1.76 GB, exceeding the 1.5 GiB weight allowance. Width/vocabulary pruning requires an adapter and quality recovery. |
| Xing 29B | Nonexpert tensors exceed 4 GB. Expert-only removal cannot meet the budget; MLA, hyper-connections and prediction layers require a separate adapter. |
| Ternary Bonsai | Custom quantization type 143 and Hadamard tensor-name metadata need the matching runtime and topology adapter. Conventional requantization is not performed. |
| Flash Next 177B | The first shard is not a complete model. Its nonexpert tensors already exceed the budget, and the second shard adds about 28.8 GB including additional architecture data. |

The inherited MiMo pruning recipe is not an interchangeable adapter for these
architectures. It also requires PyTorch and full-precision weights, which cannot
be processed within this machine's current free-RAM budget.

## Completed validation

The output has 339 tensors, all 28 original layers, and 8 experts per layer.
Its SHA256 is `4eed7d5e0915d34fd789fc3ba86e431b8f38196dc486253c84e4c71f7409092e`.
Independent structural/hash validation passed; pruning tests and script syntax
checks passed. Retained weight downloads totaled 1,595,862,016 bytes rather than
downloading the full 8,071,295,264-byte source.

The CPU inference guard measured 812,199,936 bytes of free physical RAM and
declined model loading because it requires 2 GiB free RAM. No Mellum inference
or coding-quality result is claimed. See `reports/mellum-validation.json`,
`reports/mellum-pruning.json`, and `reports/mellum-smoke.json`.
