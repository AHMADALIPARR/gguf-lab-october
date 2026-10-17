#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Build corpus shard v4 (pattern: build_corpus_snapkitty.py):
  - SNAPKITTYWEST/TWIN       clone /workspace/TWIN
  - SNAPKITTYWEST/ai-free    clone /workspace/ai-free (master; Lean 4)
  - AHMADALIPARR/legacy-logic-archive: its own fetch-corpus.pl -> normalize.pl ->
    export-training.pl allowlist (/workspace/lla-work/export), first converted
    to legacy.jsonl with a `text` field, then read here.
Harness only. Exact-dup drop (sha256 of stripped text), measured token counts.
Outputs pytorch-4b/data/v4/: corpus.jsonl, tokens.bin, val.bin, stats.json.
"""
import hashlib, json, os, pathlib, sys
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_corpus_snapkitty as B  # reuse extract(), TEXT, SKIP_* lists

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "pytorch-4b/data/v4"
LLA = pathlib.Path("/workspace/lla-work/export")
SOURCES = [("TWIN", pathlib.Path("/workspace/TWIN")), ("ai-free", pathlib.Path("/workspace/ai-free"))]
TEXT = B.TEXT | {".scala", ".thrift", ".proto", ".erl", ".hrl", ".exs", ".gradle", ".bzl", ".lean"}
PRIORITY = {"ai-free": 2}                 # Lean 4 repeated in train split only
CAP_TOKENS = {"legacy-logic-archive": 4_000_000}   # OTP 18 export is large; subsample
VAL_FRAC = 0.02


def legacy_jsonl():
    """ALLOWLIST.jsonl + sha256/ blobs -> legacy.jsonl {"text","source","license"}."""
    out = LLA / "legacy.jsonl"
    with open(out, "w") as f:
        for l in open(LLA / "ALLOWLIST.jsonl"):
            r = json.loads(l)
            raw = (LLA / "sha256" / r["sha256"]).read_bytes()
            if b"\x00" in raw[:4096] or len(raw) > B.MAX_FILE: continue
            f.write(json.dumps({"text": raw.decode("utf-8", "replace"), "source": r["id"],
                                "license": r["license"]}, ensure_ascii=False) + "\n")
    return out


def docs():
    for tag, base in SOURCES:
        for dp, dns, fns in os.walk(base):
            dns[:] = sorted(d for d in dns if d not in B.SKIP_DIRS and d not in B.SKIP_DIRS_EXTRA)
            for fn in sorted(fns):
                p = pathlib.Path(dp) / fn; ext = p.suffix.lower()
                if ext not in TEXT or fn in B.SKIP_NAMES or p.is_symlink() or fn.endswith(".lock"): continue
                if ext == ".py" and int(hashlib.sha256(str(p).encode()).hexdigest(), 16) % 100 >= B.PY_KEEP * 100:
                    yield tag, None, "py_downweight"; continue
                if ext != ".pdf" and p.stat().st_size > B.MAX_FILE:
                    yield tag, None, "too_big"; continue
                try: texts = B.extract(p)
                except Exception: yield tag, None, "error"; continue
                for t in texts: yield tag, (t, f"{tag}:{p.relative_to(base)}"), None
    order = [json.loads(l) for l in open(legacy_jsonl())]
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(order)):   # random subsample under the cap
        yield "legacy-logic-archive", (order[i]["text"], "legacy-logic-archive:" + order[i]["source"]), None


def main():
    tok = Tokenizer.from_file(str(ROOT / "pytorch-4b/tokenizer/tokenizer.json")); eos = tok.token_to_id("</s>")
    OUT.mkdir(parents=True, exist_ok=True)
    seen, ids, per, skipped = set(), [], {}, {}
    with open(OUT / "corpus.jsonl", "w") as f:
        for tag, d, why in docs():
            if why: skipped[why] = skipped.get(why, 0) + 1; continue
            t, src = d; t = t.strip()
            if len(t) < 64: skipped["empty"] = skipped.get("empty", 0) + 1; continue
            h = hashlib.sha256(t.encode()).hexdigest()
            if h in seen: skipped["dup"] = skipped.get("dup", 0) + 1; continue
            if tag in CAP_TOKENS and per.get(tag, {}).get("tokens", 0) >= CAP_TOKENS[tag]:
                skipped["cap"] = skipped.get("cap", 0) + 1; continue
            seen.add(h)
            n = tok.encode(t).ids
            f.write(json.dumps({"text": t, "source": src}, ensure_ascii=False) + "\n")
            ids.append((n + [eos], PRIORITY.get(tag, 1), tag))
            s = per.setdefault(tag, {"docs": 0, "tokens": 0, "bytes": 0})
            s["docs"] += 1; s["tokens"] += len(n) + 1; s["bytes"] += len(t.encode())
    rng = np.random.default_rng(0); order = rng.permutation(len(ids))
    nval = int(len(ids) * VAL_FRAC); val, tr = [], []
    trn = {}
    for j, i in enumerate(order):
        d, rep, tag = ids[i]
        if j < nval: val.extend(d)
        else: tr.extend(d * rep); trn[tag] = trn.get(tag, 0) + len(d) * rep
    np.array(val, dtype=np.uint16).tofile(OUT / "val.bin")
    arr = np.array(tr, dtype=np.uint16); arr.tofile(OUT / "tokens.bin")
    stats = {"docs": len(ids), "train_tokens": int(arr.size), "val_tokens": len(val),
             "unique_tokens": sum(len(d) for d, _, _ in ids), "per_source_unique": per,
             "per_source_train_tokens_after_repeat": trn, "skipped": skipped}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
