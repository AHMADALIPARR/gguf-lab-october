#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Build the Snapkitty corpus: plain-text training material from
  - Hugging Face org Snapkitty (models, datasets, spaces; text files only,
    no weight blobs), fetched into /workspace/snapkitty-hf/repos and
    /workspace/snapkitty-hf/<dataset> (hf download),
  - Zenodo records with creator "PARR, AHMAD" (CC-BY-4.0 PDFs, pdftotext),
  - GitHub SNAPKITTYWEST repos cloned into /workspace/snapkitty-gh.
Harness code only. Every document gets its source path; exact duplicates
(SHA-256 of stripped text) are dropped. Token counts are measured with the
repo tokenizer, never estimated.
Outputs (pytorch-4b/data/snapkitty/): corpus.jsonl, tokens.bin (uint16 ids,
EOS after each doc), stats.json.
"""
import hashlib, json, os, pathlib, subprocess, sys
import numpy as np
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "pytorch-4b/data/snapkitty2"
SOURCES = [("hf", pathlib.Path("/workspace/snapkitty-hf/repos")),
           ("hf-dataset", pathlib.Path("/workspace/snapkitty-hf")),
           ("zenodo", pathlib.Path("/workspace/snapkitty-hf/zenodo/pdf")),
           ("github", pathlib.Path("/workspace/snapkitty-gh"))]
CLONES = ["FEDERATED-SKUNKSLAB", "worm-modula2", "sovereign-mimo-4b", "nn16", "sovereign-mum", "Common-Metadata-Repository", "sovereign-mcp",
          "math-skills", "math-engine", "ortho32-mcp", "RBG-ibm-meta-corpus", "mathlib5-spec-mcp", "vacuum-mojo-stack"]
SOURCES += [("clone", pathlib.Path("/workspace") / c) for c in CLONES]
# Priority sources are repeated in the train split only (never in val).
PRIORITY = {"clone:FEDERATED-SKUNKSLAB": 3, "clone:mathlib5-spec-mcp": 3, "clone:sovereign-mimo-4b": 3}
PRIORITY_PATHS = {"Common-Metadata-Repository/TRAINING/": 3}   # the repo's own curated training files
VAL_FRAC = 0.02
# Priority clones are walked first so dedup keeps their copy (and its repeat weight).
SOURCES.sort(key=lambda ts: 0 if f"{ts[0]}:{ts[1].name}" in PRIORITY else 1)
PY_KEEP = 0.2            # Python is harness code: keep ~20% of .py files (hash-deterministic)
CAP_TOKENS = {"clone:Common-Metadata-Repository": 1_500_000}  # subsample so it does not dominate
SKIP_DIRS_EXTRA = {"vendor", "third_party", "external", "deps", ".github", "site-packages", "fixtures"}
TEXT = {".md", ".txt", ".tex", ".lean", ".py", ".rs", ".rb", ".pl", ".ecl", ".oz", ".scm", ".js", ".mjs", ".ts",
        ".toml", ".yaml", ".yml", ".bib", ".c", ".h", ".cpp", ".hpp", ".hs", ".ml", ".go", ".sh", ".rst", ".org",
        ".v", ".agda", ".clj", ".ex", ".idr", ".asm", ".s", ".adb", ".ads", ".cob", ".cbl", ".f", ".fs", ".lisp",
        ".el", ".jl", ".mojo", ".zig", ".nim", ".sql", ".cs", ".java", ".kt", ".swift", ".m", ".xml", ".xsl",
        ".apl", ".i", ".cu", ".sv", ".pas", ".bas", ".mod", ".def", ".cts", ".html", ".css", ".tsx", ".jsx", ".json", ".jsonl", ".csv", ".pdf", ".ipynb"}
SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "_crawl", ".cache",
             "zenodo"}
SKIP_NAMES = {"Cargo.lock", "package-lock.json", "yarn.lock", "tokenizer.json", "vocab.json", "merges.txt",
              "pnpm-lock.yaml", "poetry.lock"}
MAX_FILE = 2 * 1024 * 1024   # skip larger text files (data dumps, logs)
JSON_KEYS = ("text", "content", "instruction", "input", "output", "response", "prompt", "completion", "body",
             "abstract", "question", "answer", "chapter", "title", "description")


def strings_from_json(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and k.lower() in JSON_KEYS and len(v) > 40:
                out.append(v)
            elif isinstance(v, (dict, list)):
                strings_from_json(v, out)
    elif isinstance(o, list):
        for v in o:
            strings_from_json(v, out)


def extract(p: pathlib.Path):
    ext = p.suffix.lower()
    if ext == ".pdf":
        r = subprocess.run(["pdftotext", "-q", "-layout", str(p), "-"], capture_output=True, timeout=300)
        return [r.stdout.decode("utf-8", "replace")]
    raw = p.read_bytes()
    if b"\x00" in raw[:4096]:
        return []
    t = raw.decode("utf-8", "replace")
    if ext == ".jsonl":
        out = []
        for line in t.splitlines():
            try:
                strings_from_json(json.loads(line), out)
            except Exception:
                pass
        return ["\n\n".join(out)] if out else []
    if ext in (".json", ".ipynb"):
        try:
            o = json.loads(t)
        except Exception:
            return []
        out = []
        if ext == ".ipynb":
            for c in o.get("cells", []):
                out.append("".join(c.get("source", [])))
        else:
            strings_from_json(o, out)
        return ["\n\n".join(out)] if out else []
    return [t]


def main():
    tok = Tokenizer.from_file(str(ROOT / "pytorch-4b/tokenizer/tokenizer.json"))
    eos = tok.token_to_id("</s>")
    OUT.mkdir(parents=True, exist_ok=True)
    seen, ids, docs = set(), [], 0
    per = {}
    skipped = {"too_big": 0, "dup": 0, "empty": 0, "error": 0}
    with open(OUT / "corpus.jsonl", "w") as f:
        for tag, base in SOURCES:
            if not base.exists():
                continue
            for dp, dns, fns in os.walk(base):
                dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and d not in SKIP_DIRS_EXTRA and not (tag == "hf-dataset" and dp == str(base) and d == "repos"))
                for fn in sorted(fns):
                    p = pathlib.Path(dp) / fn
                    if p.suffix.lower() not in TEXT or fn in SKIP_NAMES or p.is_symlink() or fn.endswith(".lock"):
                        continue
                    if p.suffix.lower() == ".py" and int(hashlib.sha256(str(p).encode()).hexdigest(), 16) % 100 >= PY_KEEP * 100:
                        skipped["py_downweight"] = skipped.get("py_downweight", 0) + 1; continue
                    if p.suffix.lower() != ".pdf" and p.stat().st_size > MAX_FILE:
                        skipped["too_big"] += 1; continue
                    try:
                        texts = extract(p)
                    except Exception:
                        skipped["error"] += 1; continue
                    for t in texts:
                        t = t.strip()
                        if len(t) < 64:
                            skipped["empty"] += 1; continue
                        h = hashlib.sha256(t.encode()).hexdigest()
                        if h in seen:
                            skipped["dup"] += 1; continue
                        seen.add(h)
                        rel = str(p.relative_to(base))
                        repo = base.name if tag == "clone" else "/".join(rel.split("/")[:3 if tag == "hf" else 1])
                        ck = f"{tag}:{repo}"
                        if ck in CAP_TOKENS and per.get(ck, {}).get("tokens", 0) >= CAP_TOKENS[ck]:
                            skipped["cap"] = skipped.get("cap", 0) + 1; continue
                        f.write(json.dumps({"text": t, "source": f"{tag}:{rel}"}, ensure_ascii=False) + "\n")
                        n = tok.encode(t).ids
                        rep = PRIORITY.get(ck, 1)
                        if tag == "clone" and any(f"{base.name}/{rel}".startswith(q) for q in PRIORITY_PATHS):
                            rep = 3
                        ids.append((n + [eos], rep)); docs += 1
                        key = f"{tag}:{repo}"
                        s = per.setdefault(key, {"docs": 0, "tokens": 0, "bytes": 0})
                        s["docs"] += 1; s["tokens"] += len(n) + 1; s["bytes"] += len(t.encode())
    rng = np.random.default_rng(0); order = rng.permutation(len(ids))
    nval = int(len(ids) * VAL_FRAC); val, tr = [], []
    for j, i in enumerate(order):
        d, rep = ids[i]
        if j < nval: val.extend(d)
        else: tr.extend(d * rep)
    np.array(val, dtype=np.uint16).tofile(OUT / "val.bin")
    arr = np.array(tr, dtype=np.uint16)
    arr.tofile(OUT / "tokens.bin")
    by_tag = {}
    for k, v in per.items():
        b = by_tag.setdefault(k.split(":")[0], {"docs": 0, "tokens": 0, "bytes": 0, "repos": 0})
        for x in ("docs", "tokens", "bytes"):
            b[x] += v[x]
        b["repos"] += 1
    stats = {"docs": docs, "tokens": int(arr.size), "val_tokens": len(val), "unique_tokens": sum(len(d) for d, _ in ids), "by_source": by_tag, "skipped": skipped,
             "per_repo": dict(sorted(per.items(), key=lambda kv: -kv[1]["tokens"]))}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps({k: stats[k] for k in ("docs", "tokens", "val_tokens", "unique_tokens", "by_source", "skipped")}, indent=1))


if __name__ == "__main__":
    main()
