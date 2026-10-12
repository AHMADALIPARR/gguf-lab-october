<!-- Copyright (C) 2026 Ahmad Ali Parr; SPDX-License-Identifier: AGPL-3.0-only -->
# Hugging Face artifact upload

Two model-specific repositories avoid mixing a text-generation architecture and
an embedding architecture in one inference repository. By default the namespace
comes from the token's authenticated account:

- `<account>/Mellum2.1-E8-experimental-GGUF`
- `<account>/EmbeddingGemma2-Q4-GGUF`

Store a token with write access in the ignored `.local/hf-token` file, or supply
`HF_TOKEN` in the local process environment. Tokens are never command-line
arguments, printed output or uploaded files. `.local/`, `models/`, `.venv/` and
SDK caches are excluded from Git. The SDK cache stays inside the workspace.

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r publish/requirements.txt
.venv/Scripts/python.exe scripts/upload_hf.py --auth-check
.venv/Scripts/python.exe scripts/upload_hf.py
```

Use `--namespace organization-name` for an authorized organization destination.
The publisher verifies local sizes/hashes before creating repositories, uses one
upload worker with Xet high-performance mode disabled, uploads only explicit
artifact/metadata paths, and checks remote LFS SHA256 and size after each commit.
It refuses to replace a different existing model artifact. The transfer uses no
GPU. Results go in `reports/hf-upload.json`.

Mellum remains explicitly experimental and quality-unvalidated. The embedding
artifact is an unchanged verified mirror. Model cards, Apache-2.0 license text,
attribution, checksums and validation/provenance accompany the weights.
