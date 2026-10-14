#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare token ids from tokenizer/tokenizer.json (HF) with llama-tokenize on a GGUF.

Usage: tokenizer_parity.py MODEL.gguf [--llama-bin DIR]
Test suite: DSS prompts from test_balanced.jsonl, numbers, code, Markdown,
unicode, whitespace. Prints match counts and the first mismatches.
"""
import argparse, json, pathlib, subprocess, sys
from tokenizers import Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
EXTRA = [
    "0", "7", "42", "123456789", "3.14159", "-17", "1e-6", "2^31-1", "x=10,y=20", "1,000,000",
    "bound k 40 a 801 -> b=800; 801>=800 yes", "is DSS 3 5 6 -> no 6=1+5",
    "def f(x):\n    return x**2 + 1\n", "for (int i = 0; i < n; i++) { s += a[i]; }",
    "import numpy as np\nnp.zeros((3, 4), dtype=np.int16)", "    \t  indented\n\n\nlines  ",
    "# Title\n\n- item **bold** and `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |",
    "[link](https://example.com/path?q=1&r=2)", "> quote\n```python\nprint('hi')\n```",
    "café naïve résumé", "Straße über Ärger", "日本語のテキスト", "中文测试 123", "emoji 🙂👍🏽",
    "Ελληνικά και русский", "العربية", "tab\tsep\tvalues", "it's they're we've I'm you'll he'd",
    "  leading spaces", "trailing spaces   ", "CRLF\r\nline", "\u00a0nbsp\u2009thin",
]


def llama_ids(binp, model, text):
    out = subprocess.run([str(binp / "llama-tokenize"), "-m", model, "-p", text, "--ids",
                          "--no-bos", "--log-disable"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gguf")
    ap.add_argument("--llama-bin", default="/workspace/llama.cpp/build/bin")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(HERE.parent / "tokenizer/tokenizer.json"))
    rows = [json.loads(l) for l in open(HERE / "data/test_balanced.jsonl")]
    dss = [r["prompt"] + r["answer"] for r in rows]
    suites = {"dss": dss, "other": EXTRA}
    res = {}
    for name, texts in suites.items():
        ok, bad = 0, []
        for t in texts:
            h = tok.encode(t).ids; l = llama_ids(pathlib.Path(a.llama_bin), a.gguf, t)
            if h == l: ok += 1
            else: bad.append({"text": t, "hf": h, "llama": l})
        res[name] = {"match": ok, "total": len(texts), "mismatches": bad[:5]}
        print(f"{name}: {ok}/{len(texts)} identical")
        for b in bad[:5]: print("  MISMATCH", repr(b["text"]), b["hf"], b["llama"])
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
