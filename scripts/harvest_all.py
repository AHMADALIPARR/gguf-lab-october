#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Mass-harvest RE training corpus from all workspace repos.

Walks ~/workspace/*/ collecting text files (code, docs, specs) into
a unified JSONL corpus. Deduplicates by content hash. Skips binaries,
vendored deps, and build artifacts.

Usage:
    python3 scripts/harvest_all.py --out pytorch-4b/data/full-corpus.jsonl
"""
import argparse
import hashlib
import json
import pathlib

WORKSPACE = pathlib.Path.home() / "workspace"

# File types to harvest
EXTENSIONS = {
    ".md", ".txt", ".py", ".c", ".h", ".rs", ".js", ".ts",
    ".java", ".go", ".cob", ".asm", ".s", ".als", ".json",
    ".yaml", ".yml", ".toml", ".pl", ".adb", ".ads",
}

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "target", "__pycache__", ".venv",
    "assets", "vendor", "dist", "build", ".next",
}

# Skip files over 5MB (likely generated)
MAX_FILE_BYTES = 5 * 1024 * 1024


def should_skip(path):
    parts = path.parts
    return any(skip in parts for skip in SKIP_DIRS)


def harvest():
    seen_hashes = set()
    docs = []
    skipped = 0

    for repo_dir in sorted(WORKSPACE.iterdir()):
        if not repo_dir.is_dir():
            continue
        if repo_dir.name.startswith(".") or repo_dir.name.startswith("_"):
            continue

        for path in repo_dir.rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path):
                continue
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
                if size > MAX_FILE_BYTES or size < 100:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if len(text.strip()) < 100:
                continue

            # Deduplicate by content
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(h)

            docs.append({
                "source": str(path.relative_to(WORKSPACE)),
                "chars": len(text),
                "text": text,
            })

    return docs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    print("Harvesting...", flush=True)
    docs, skipped = harvest()
    total_chars = sum(d["chars"] for d in docs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({
                "text": d["text"], "source": d["source"]}) + "\n")

    print(f"Documents: {len(docs):,}")
    print(f"Dedup skipped: {skipped:,}")
    print(f"Total chars: {total_chars:,}")
    print(f"Est. tokens: {total_chars // 4:,}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
