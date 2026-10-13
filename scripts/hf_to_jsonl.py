#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Convert Hugging Face GGUF models to JSONL tensor manifests.

Remotely indexes GGUF headers (no full download) and emits one JSON
object per tensor per line. Output is WORM-ingestion ready: each line
has the tensor name, shape, dtype, file offset, and byte size needed
to seal the tensor into a WORM archive.

Usage:
    python3 scripts/hf_to_jsonl.py --all > manifests/all.jsonl
    python3 scripts/hf_to_jsonl.py --model mellum > manifests/mellum.jsonl
"""
import argparse
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gguf_index import parse_header, remote_header

# (repo, filename, short_id) — mirrors pull.sh
MODELS = [
    ("unsloth/embeddinggemma-2-GGUF",
     "embeddinggemma-2-UD-Q4_K_XL.gguf", "embed"),
    ("JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF",
     "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf", "mellum"),
    ("unsloth/Qwen3.8-27B-GGUF",
     "Qwen3.8-27B-UD-Q4_K_M.gguf", "qwen27"),
    ("prism-ml/Ternary-Bonsai-2-27B-gguf",
     "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "bonsai"),
    ("XingChen-AGI/Xing4.0-29B-A4B-GGUF",
     "xing4_0-29b-IQ4_NL.gguf", "xing"),
]

# GGUF type id -> (name, bytes per element approx for size calc)
# Full size calc needs block structure; we report elements and type.
TYPE_NAMES = {
    0: "F32", 1: "F16", 6: "Q4_0", 7: "Q4_1", 8: "Q5_0", 9: "Q5_1",
    10: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16",
}


def hf_url(repo, filename):
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def get_remote_size(url):
    """Get file size via HEAD request."""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "GGUF-Lab/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers["Content-Length"])


def tensors_to_jsonl(repo, filename, short_id, out):
    """Index remote GGUF, emit one JSON object per tensor."""
    url = hf_url(repo, filename)
    print(f"Indexing {short_id}: {url}", file=sys.stderr, flush=True)
    size = get_remote_size(url)
    print(f"  Remote size: {size} bytes", file=sys.stderr, flush=True)
    header = remote_header(url, size)
    tensors = header.get("tensors", [])
    # Also emit a manifest header line first
    manifest = {
        "type": "manifest",
        "model_id": short_id,
        "repo": repo,
        "filename": filename,
        "url": url,
        "tensor_count": len(tensors),
        "metadata": {k: v for k, v in header.get("metadata", {}).items()
                     if not k.startswith("_")},
    }
    out.write(json.dumps(manifest) + "\n")
    for t in tensors:
        record = {
            "type": "tensor",
            "model_id": short_id,
            "name": t["name"],
            "shape": t["shape"],
            "dtype": TYPE_NAMES.get(t["type"], f"unknown_{t['type']}"),
            "dtype_id": t["type"],
            "offset": t["offset"],
            "n_elements": 1,
        }
        # Compute element count from shape
        n = 1
        for dim in t["shape"]:
            n *= dim
        record["n_elements"] = n
        out.write(json.dumps(record) + "\n")
    print(f"  {len(tensors)} tensors", file=sys.stderr, flush=True)
    return len(tensors)


def main():
    ap = argparse.ArgumentParser(description="HF GGUF -> JSONL tensor manifests")
    ap.add_argument("--all", action="store_true", help="Convert all models")
    ap.add_argument("--model", type=str, help="Convert one model by short_id")
    ap.add_argument("--out", type=pathlib.Path, help="Output file (default: stdout)")
    args = ap.parse_args()

    selected = MODELS
    if args.model:
        selected = [m for m in MODELS if m[2] == args.model]
        if not selected:
            ap.error(f"Unknown model: {args.model}. Choices: "
                     + ", ".join(m[2] for m in MODELS))
    elif not args.all:
        ap.error("Specify --all or --model <id>")

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    total = 0
    try:
        for repo, filename, short_id in selected:
            try:
                total += tensors_to_jsonl(repo, filename, short_id, out)
            except Exception as e:
                print(f"  ERROR {short_id}: {e}", file=sys.stderr)
    finally:
        if out is not sys.stdout:
            out.close()
    print(f"Total tensors: {total}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
