#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Merge multiple GGUF tensor manifests into one unified model manifest.

Takes the per-model JSONL manifests (from hf_to_jsonl.py) and combines
them into a single tensor inventory — one logical model composed of
expert sub-models. Tensor names are prefixed with the model_id to avoid
collisions (e.g. "mellum/blk.0.attn.weight").

This is the manifest-level equivalent of hologram-re's `strip` tool:
checkpoint(s) -> unified JSON tensor manifest. The actual tensor bytes
are not merged here — this produces the inventory that a WORM materializer
or MoE router consumes to treat the ensemble as one model.

Usage:
    python3 scripts/merge_models.py --out manifests/unified.jsonl
    python3 scripts/merge_models.py --models mellum qwen27 --out manifests/coding.jsonl
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"


def merge_manifests(model_ids, out_path):
    """Merge per-model JSONL manifests into one unified manifest."""
    all_tensors = []
    sources = []

    for model_id in model_ids:
        mp = MANIFEST_DIR / f"{model_id}.jsonl"
        if not mp.exists():
            print(f"Warning: {mp} not found, skipping", file=sys.stderr)
            continue

        manifest = None
        tensors = []
        with open(mp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "manifest":
                    manifest = record
                elif record.get("type") == "tensor":
                    tensors.append(record)

        if manifest is None:
            print(f"Warning: {mp} has no manifest line, skipping",
                  file=sys.stderr)
            continue

        sources.append({
            "model_id": model_id,
            "repo": manifest.get("repo"),
            "filename": manifest.get("filename"),
            "tensor_count": len(tensors),
        })

        # Prefix tensor names with model_id to avoid collisions
        for t in tensors:
            unified = dict(t)
            unified["name"] = f"{model_id}/{t['name']}"
            unified["source_model"] = model_id
            all_tensors.append(unified)

        print(f"  {model_id}: {len(tensors)} tensors", file=sys.stderr)

    # Sort for deterministic output
    all_tensors.sort(key=lambda t: t["name"])

    # Write unified manifest
    with open(out_path, "w", encoding="utf-8") as f:
        # Unified header
        f.write(json.dumps({
            "type": "unified_manifest",
            "version": "1.0",
            "model_count": len(sources),
            "tensor_count": len(all_tensors),
            "sources": sources,
        }) + "\n")
        # All tensors
        for t in all_tensors:
            f.write(json.dumps(t) + "\n")

    print(f"Merged {len(all_tensors)} tensors from {len(sources)} models -> {out_path}",
          file=sys.stderr)
    return len(all_tensors)


def main():
    ap = argparse.ArgumentParser(
        description="Merge GGUF manifests into one unified model")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model IDs to merge (default: all in manifests/)")
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="Output unified JSONL manifest")
    args = ap.parse_args()

    if args.models is None:
        # Auto-discover: all manifests except unified ones
        args.models = sorted(
            p.stem for p in MANIFEST_DIR.glob("*.jsonl")
            if not p.stem.startswith("unified")
        )
        print(f"Auto-discovered models: {args.models}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = merge_manifests(args.models, args.out)
    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
