#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Sandboxed benchmark runner for pruned GGUF models.

Isolation guarantees:
- Runs each case as a separate subprocess with timeout
- Writes outputs only to sandbox/outputs/ (gitignored)
- Validates model SHA256 before running (if configured)
- Fails closed: any missing prerequisite aborts before inference
- Never modifies model files or repo state

Usage:
    python3 sandbox/run.py --runtime /path/to/llama-cli --model models/mellum-E8-experimental.gguf
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "sandbox"
CONFIG = SANDBOX / "config.json"
INPUTS = SANDBOX / "inputs" / "prompts.json"
OUTPUTS = SANDBOX / "outputs"


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def check_ram(required_bytes):
    """Return (ok, available_bytes)."""
    if os.name == "nt":
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in
                ["total_phys", "available_phys", "total_page", "available_page",
                 "total_virtual", "available_virtual", "available_extended"]]
        st = MS()
        st.length = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.available_phys >= required_bytes, st.available_phys
    avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return avail >= required_bytes, avail


def sha256_file(path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_case(runtime, model, cfg, case):
    """Run one prompt in a sandboxed subprocess. Returns result dict."""
    rt = cfg["runtime"]
    lim = cfg["resource_limits"]
    cmd = [
        str(runtime), "-m", str(model),
        "-ngl", str(rt["gpu_layers"]),
        "-t", str(rt["threads"]), "-tb", str(rt["threads_batch"]),
        "-c", str(rt["context_size"]),
        "-b", str(rt["batch_size"]), "-ub", str(rt["ubatch_size"]),
        "-n", str(rt["max_tokens"]),
        "--single-turn", "--no-warmup",
        "-p", case["prompt"],
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=lim["per_case_timeout_seconds"],
        )
        elapsed = time.time() - start
        out = proc.stdout[-lim["max_output_chars"]:]
        passed = all(s in out for s in case["must_contain"])
        # Syntax check: try compiling extracted code
        syntax_ok = False
        code = out
        if "```" in code:
            for part in code.split("```"):
                if "def " in part:
                    code = part.replace("python", "", 1).strip()
                    break
        try:
            compile(code, "<gen>", "exec")
            syntax_ok = True
        except SyntaxError:
            pass
        return {
            "id": case["id"], "passed": passed, "syntax_valid": syntax_ok,
            "elapsed_seconds": round(elapsed, 1), "exit_code": proc.returncode,
            "output_chars": len(proc.stdout),
            "stderr_tail": proc.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        return {"id": case["id"], "passed": False, "timeout": True,
                "elapsed_seconds": lim["per_case_timeout_seconds"]}


def main():
    ap = argparse.ArgumentParser(description="Sandboxed GGUF benchmark runner")
    ap.add_argument("--runtime", type=pathlib.Path, required=True,
                    help="Path to llama-cli executable")
    ap.add_argument("--model", type=pathlib.Path, required=True,
                    help="Path to GGUF model file")
    ap.add_argument("--expected-sha256", type=str, default=None,
                    help="Expected SHA256 of model file (validates before run)")
    ap.add_argument("--tag", type=str, default="benchmark",
                    help="Tag for output report filename")
    args = ap.parse_args()

    cfg = load_config()
    lim = cfg["resource_limits"]
    report = {
        "sandbox_version": cfg["sandbox_version"],
        "model": str(args.model),
        "runtime": str(args.runtime),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Fail closed: prerequisites
    if not args.runtime.is_file():
        report.update(status="blocked", reason="runtime_missing")
        print(json.dumps(report, indent=2))
        return 2
    if not args.model.is_file():
        report.update(status="blocked", reason="model_missing")
        print(json.dumps(report, indent=2))
        return 2
    if args.expected_sha256:
        actual = sha256_file(args.model)
        if actual != args.expected_sha256.lower():
            report.update(status="blocked", reason="sha256_mismatch",
                          expected=args.expected_sha256, actual=actual)
            print(json.dumps(report, indent=2))
            return 2
        report["sha256_verified"] = True
    ok, avail = check_ram(lim["max_ram_bytes"])
    report["free_ram_bytes"] = avail
    if not ok:
        report.update(status="blocked", reason="insufficient_ram",
                      required_bytes=lim["max_ram_bytes"])
        print(json.dumps(report, indent=2))
        return 3

    # Run cases
    cases = json.loads(INPUTS.read_text(encoding="utf-8"))
    results = []
    deadline = time.time() + lim["total_timeout_seconds"]
    for case in cases:
        if time.time() > deadline:
            results.append({"id": case["id"], "passed": False,
                            "reason": "total_timeout_exceeded"})
            continue
        print(f"[{case['id']}] running...", flush=True)
        results.append(run_case(args.runtime.resolve(),
                               args.model.resolve(), cfg, case))

    n_pass = sum(1 for r in results if r.get("passed"))
    n_syn = sum(1 for r in results if r.get("syntax_valid"))
    report.update(
        status="completed",
        cases_run=len(results),
        cases_passed=n_pass,
        syntax_valid_count=n_syn,
        scope="Hand-written prompts with substring + syntax checks. "
              "NOT HumanEval/MBPP. Substring passing does not prove competence.",
        inference_validated=n_pass == len(results),
        coding_quality_validated=False,
        results=results,
    )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"{args.tag}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {out_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
