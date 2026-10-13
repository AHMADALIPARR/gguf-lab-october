#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""WORM metadata training pipeline.

Takes JSONL tensor manifests (from hf_to_jsonl.py) and trains them into
WORM-sealed weight archives. "Training" here means: validate the metadata,
compute the WORM layout, hash-chain every tensor into a seal manifest,
and emit a sealed archive descriptor ready for the evoke pipeline.

Pipeline stages:
  1. INGEST   - read JSONL manifest, validate schema per line
  2. VALIDATE - check tensor metadata completeness (name, shape, dtype, offset)
  3. LAYOUT   - compute WORM staging layout (64-byte aligned offsets)
  4. SEAL     - hash-chain tensors into a SealManifest (each seal commits
                to the previous hash, like the WORM chain architecture)
  5. EMIT     - write sealed WORM archive descriptor as JSONL

The output is a WORM archive descriptor: it does not contain weight bytes,
only the sealed metadata needed to materialize the archive when the
source GGUF is available. This matches the glm-evoke two-phase ingest:
phase 1 validates metadata with zero weight allocation.

Usage:
    python3 scripts/worm_train.py --manifest manifests/mellum.jsonl \
        --out worm/mellum.worm.jsonl
    python3 scripts/worm_train.py --all --out-dir worm/
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# WORM staging alignment (matches glm-evoke/src/worm.rs STAGING_ALIGN)
STAGING_ALIGN = 64

# Fallback bytes-per-element estimates. Used only when a manifest lacks
# provenance fields (file_bytes, tensor_data_offset) for measurement
# from GGUF offset deltas. Quantized types use block sizes; these are
# conservative estimates.
DTYPE_BYTES = {
    "F32": 4, "F16": 2, "BF16": 2, "F64": 8,
    "Q4_0": 0.5625, "Q4_1": 0.625, "Q5_0": 0.6875, "Q5_1": 0.75,
    "Q8_0": 1.0625, "Q4_K": 0.5625, "Q5_K": 0.625, "Q6_K": 0.75,
    "Q8_K": 1.0625, "IQ4_NL": 0.5625, "IQ2_XXS": 0.3125,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
}


def align_up(v, align=STAGING_ALIGN):
    return (v + align - 1) & ~(align - 1)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_ingest(manifest_path):
    """Stage 1: INGEST - read and schema-validate JSONL manifest."""
    tensors = []
    manifest = None
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{manifest_path}:{lineno}: invalid JSON: {e}")
            rtype = record.get("type")
            if rtype == "manifest":
                if manifest is not None:
                    raise ValueError("Duplicate manifest line")
                manifest = record
            elif rtype == "tensor":
                tensors.append((lineno, record))
            else:
                raise ValueError(
                    f"{manifest_path}:{lineno}: unknown record type {rtype!r}")
    if manifest is None:
        raise ValueError(f"{manifest_path}: missing manifest line")
    return manifest, tensors


def stage_validate(manifest, tensors, manifest_path):
    """Stage 2: VALIDATE - check tensor metadata completeness."""
    errors = []
    seen_names = set()
    required = ["name", "shape", "dtype", "offset", "n_elements"]
    for lineno, t in tensors:
        for field in required:
            if field not in t:
                errors.append(f"line {lineno}: missing field {field!r}")
        name = t.get("name", "")
        if name in seen_names:
            errors.append(f"line {lineno}: duplicate tensor {name!r}")
        seen_names.add(name)
        shape = t.get("shape", [])
        if not isinstance(shape, list) or not shape:
            errors.append(f"line {lineno}: invalid shape for {name!r}")
    if errors:
        raise ValueError(
            f"{manifest_path}: validation failed:\n" + "\n".join(errors))
    # Cross-check tensor count against manifest
    expected = manifest.get("tensor_count")
    if expected is not None and expected != len(tensors):
        raise ValueError(
            f"{manifest_path}: tensor count mismatch: "
            f"manifest says {expected}, found {len(tensors)}")
    return True


def measured_byte_sizes(manifest, tensors):
    """Per-tensor byte sizes measured from GGUF offset deltas.

    Tensors are sorted by source offset; each tensor's measured span is
    the delta to the next tensor's offset. The last tensor's span runs
    to the end of the tensor data section
    (file_bytes - tensor_data_offset).

    Returns {name: (byte_size, size_source)} where size_source is
    "measured" or "estimated". Estimation (DTYPE_BYTES) is used only
    when the manifest lacks provenance fields or a delta is
    non-positive; overlapping offsets (negative delta) raise.
    """
    ordered = sorted(tensors, key=lambda pair: pair[1]["offset"])
    sizes = {}
    file_bytes = manifest.get("file_bytes") if manifest else None
    data_offset = manifest.get("tensor_data_offset") if manifest else None
    measurable = (
        file_bytes is not None and data_offset is not None
    )
    for i, (lineno, t) in enumerate(ordered):
        size = None
        if measurable:
            if i < len(ordered) - 1:
                delta = ordered[i + 1][1]["offset"] - t["offset"]
                # Deltas are non-negative after sorting; a zero delta
                # means a duplicate/degenerate offset, which we do not
                # trust as a measurement.
                if delta > 0:
                    size = delta
            else:
                # Last tensor: span runs to end of the data section.
                span = (file_bytes - data_offset) - t["offset"]
                if span <= 0:
                    raise ValueError(
                        f"manifest: last tensor {t.get('name')!r} "
                        f"(line {lineno}) starts at or past end of data")
                size = span
        if size is None:
            dtype = t["dtype"]
            n_elements = t["n_elements"]
            bpe = DTYPE_BYTES.get(dtype, 1.0)  # conservative fallback
            size = int(n_elements * bpe)
            sizes[t["name"]] = (size, "estimated")
        else:
            sizes[t["name"]] = (size, "measured")
    return sizes


def stage_layout(tensors, manifest=None):
    """Stage 3: LAYOUT - compute WORM staging offsets (64-byte aligned).

    Byte sizes are measured from GGUF offset deltas when the manifest
    carries provenance (file_bytes, tensor_data_offset); otherwise they
    fall back to DTYPE_BYTES estimates (flagged per tensor).
    """
    sizes = measured_byte_sizes(manifest, tensors)
    layout = []
    offset = 0
    # Preserve manifest order for the layout; sizes keyed by name.
    for _, t in tensors:
        byte_size, size_source = sizes[t["name"]]
        aligned_offset = align_up(offset)
        layout.append({
            "name": t["name"],
            "shape": t["shape"],
            "dtype": t["dtype"],
            "staging_offset": aligned_offset,
            "byte_size": byte_size,
            "size_source": size_source,
            "source_offset": t["offset"],
        })
        offset = aligned_offset + byte_size
    total = align_up(offset)
    return layout, total


def stage_seal(manifest, layout, total_bytes):
    """Stage 4: SEAL - hash-chain tensors into a SealManifest.

    Each tensor's seal commits to the previous seal hash, forming
    a tamper-evident chain. The final seal is the archive root.
    """
    model_id = manifest["model_id"]
    prev_hash = sha256_hex(f"worm-genesis:{model_id}".encode())
    seals = []
    for entry in layout:
        # Seal input: tensor identity + layout + previous hash
        seal_input = json.dumps({
            "name": entry["name"],
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "staging_offset": entry["staging_offset"],
            "byte_size": entry["byte_size"],
            "size_source": entry["size_source"],
            "prev": prev_hash,
        }, sort_keys=True).encode()
        seal_hash = sha256_hex(seal_input)
        seals.append({
            "name": entry["name"],
            "seal": seal_hash,
            "prev": prev_hash,
        })
        prev_hash = seal_hash
    # Archive root seal
    root_input = json.dumps({
        "model_id": model_id,
        "tensor_count": len(layout),
        "total_bytes": total_bytes,
        "final": prev_hash,
    }, sort_keys=True).encode()
    root_seal = sha256_hex(root_input)
    return {
        "model_id": model_id,
        "seal_chain": seals,
        "root_seal": root_seal,
        "total_bytes": total_bytes,
        "tensor_count": len(layout),
    }


def stage_emit(manifest, layout, seal_data, out_path):
    """Stage 5: EMIT - write sealed WORM archive descriptor as JSONL."""
    with open(out_path, "w", encoding="utf-8") as f:
        # Archive header
        f.write(json.dumps({
            "type": "worm_archive",
            "version": "1.0",
            "model_id": manifest["model_id"],
            "repo": manifest.get("repo"),
            "source_file": manifest.get("filename"),
            "root_seal": seal_data["root_seal"],
            "tensor_count": seal_data["tensor_count"],
            "total_bytes": seal_data["total_bytes"],
            "staging_align": STAGING_ALIGN,
        }) + "\n")
        # One line per sealed tensor
        seal_map = {s["name"]: s for s in seal_data["seal_chain"]}
        for entry in layout:
            seal = seal_map[entry["name"]]
            f.write(json.dumps({
                "type": "worm_tensor",
                "name": entry["name"],
                "shape": entry["shape"],
                "dtype": entry["dtype"],
                "staging_offset": entry["staging_offset"],
                "byte_size": entry["byte_size"],
                "size_source": entry["size_source"],
                "source_offset": entry["source_offset"],
                "seal": seal["seal"],
                "prev_seal": seal["prev"],
            }) + "\n")


def train_one(manifest_path, out_path):
    """Run the full pipeline on one manifest."""
    print(f"Training {manifest_path.name}...", file=sys.stderr)
    # Stage 1: INGEST
    manifest, tensors = stage_ingest(manifest_path)
    print(f"  ingest: {len(tensors)} tensors", file=sys.stderr)
    # Stage 2: VALIDATE
    stage_validate(manifest, tensors, manifest_path)
    print(f"  validate: OK", file=sys.stderr)
    # Stage 3: LAYOUT
    layout, total = stage_layout(tensors, manifest)
    print(f"  layout: {total} bytes staged", file=sys.stderr)
    # Stage 4: SEAL
    seal_data = stage_seal(manifest, layout, total)
    print(f"  seal: root={seal_data['root_seal'][:16]}...", file=sys.stderr)
    # Stage 5: EMIT
    stage_emit(manifest, layout, seal_data, out_path)
    print(f"  emit: {out_path}", file=sys.stderr)
    return seal_data


def main():
    ap = argparse.ArgumentParser(description="WORM metadata training pipeline")
    ap.add_argument("--manifest", type=pathlib.Path,
                    help="Input JSONL tensor manifest")
    ap.add_argument("--all", action="store_true",
                    help="Train all manifests in manifests/")
    ap.add_argument("--out", type=pathlib.Path,
                    help="Output WORM archive (for --manifest)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "worm",
                    help="Output directory (for --all)")
    args = ap.parse_args()

    if args.all:
        manifest_dir = ROOT / "manifests"
        manifests = sorted(manifest_dir.glob("*.jsonl"))
        if not manifests:
            ap.error(f"No manifests in {manifest_dir}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for mp in manifests:
            out = args.out_dir / f"{mp.stem}.worm.jsonl"
            try:
                train_one(mp, out)
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                return 1
    elif args.manifest:
        if not args.out:
            ap.error("--out required with --manifest")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        train_one(args.manifest, args.out)
    else:
        ap.error("Specify --manifest or --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
