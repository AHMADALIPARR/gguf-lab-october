#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Benchmark pruned Mellum on coding tasks. CPU-only, bounded, honest.

Runs a small set of coding prompts through llama.cpp and scores them
with deterministic checks (not LLM-judged). Reports what was measured
and what was not. Fails closed if the model or runtime is missing.
"""
import argparse
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Each case: prompt, and a list of substrings that must appear in output.
# These are smoke-level checks, not a benchmark suite like HumanEval.
CASES = [
    {
        "id": "add_function",
        "prompt": "Write a Python function add(a, b) that returns a + b. Output code only.",
        "must_contain": ["def add", "return"],
        "must_not_contain": [],
    },
    {
        "id": "fibonacci",
        "prompt": "Write a Python function fib(n) that returns the nth Fibonacci number using iteration. Output code only.",
        "must_contain": ["def fib", "for", "return"],
        "must_not_contain": [],
    },
    {
        "id": "even_check",
        "prompt": "Write a Python function is_even(n) that returns True if n is even, False otherwise. Output code only.",
        "must_contain": ["def is_even", "return"],
        "must_not_contain": [],
    },
    {
        "id": "syntax_valid",
        "prompt": "Write a Python function greet(name) that returns the string 'Hello, ' + name. Output code only.",
        "must_contain": ["def greet", "return"],
        "must_not_contain": ["Traceback", "Error"],
    },
]

def run_case(runtime, model, case, timeout=180):
    cmd = [
        str(runtime), "-m", str(model), "-ngl", "0", "-t", "1", "-tb", "1",
        "-c", "512", "-b", "32", "-ub", "32", "-n", "128",
        "--single-turn", "--no-warmup", "-p", case["prompt"],
    ]
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        elapsed = time.time() - start
        output = result.stdout
        passed = all(s in output for s in case["must_contain"])
        passed = passed and not any(s in output for s in case["must_not_contain"])
        # Try to compile the output as Python to check syntax
        syntax_ok = False
        try:
            # Extract code block if present
            code = output
            if "```" in code:
                parts = code.split("```")
                for p in parts:
                    if "def " in p:
                        code = p.replace("python", "", 1).strip()
                        break
            compile(code, "<generated>", "exec")
            syntax_ok = True
        except SyntaxError:
            pass
        return {
            "id": case["id"],
            "passed": passed,
            "syntax_valid": syntax_ok,
            "elapsed_seconds": round(elapsed, 1),
            "exit_code": result.returncode,
            "output_chars": len(output),
            "output_tail": output[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"id": case["id"], "passed": False, "syntax_valid": False,
                "elapsed_seconds": timeout, "timeout": True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=pathlib.Path, required=True,
                        help="Path to llama-cli executable")
    parser.add_argument("--model", type=pathlib.Path, required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--report", type=pathlib.Path,
                        default=ROOT / "reports/mellum-benchmark.json")
    args = parser.parse_args()

    if not args.runtime.is_file():
        raise FileNotFoundError(f"Runtime not found: {args.runtime}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    results = []
    for case in CASES:
        print(f"Running {case['id']}...", flush=True)
        results.append(run_case(args.runtime.resolve(), args.model.resolve(), case))

    passed = sum(1 for r in results if r.get("passed"))
    syntax_ok = sum(1 for r in results if r.get("syntax_valid"))
    report = {
        "model": str(args.model),
        "runtime": str(args.runtime),
        "cases_run": len(results),
        "cases_passed": passed,
        "syntax_valid_count": syntax_ok,
        "benchmark_scope": "4 hand-written coding prompts with substring checks; "
                           "NOT HumanEval, NOT MBPP, NOT a general coding benchmark",
        "inference_validated": passed == len(results),
        "coding_quality_validated": False,
        "note": "Passing substring checks does not establish coding competence. "
                "A real benchmark (HumanEval/MBPP) was not run.",
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    raise SystemExit(main())
