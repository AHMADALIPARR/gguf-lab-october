#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Sparse MoE router: embedding-driven expert selection.

Uses EmbeddingGemma to embed prompts, routes to top-k experts by cosine
similarity to pre-computed anchor centroids. Sparse: only selected experts
are loaded for inference.

Pipeline:
  1. Build centroids: embed each expert's anchor prompts, average -> centroid
  2. Route: embed query, cosine similarity to each centroid, pick top-k
  3. Inference: run selected expert(s) via llama-cli

Requires llama-server with --embedding for the router model.
"""
import argparse
import json
import math
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MOE = ROOT / "sandbox" / "moe"
EXPERTS = MOE / "experts.json"
CENTROIDS = MOE / "centroids.json"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_via_server(server_url, text):
    """Get embedding from llama-server /embedding endpoint."""
    data = json.dumps({"content": text}).encode()
    req = urllib.request.Request(
        server_url.rstrip("/") + "/embedding",
        data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    # llama-server returns {"embedding": [...]} or {"data": [{"embedding": [...]}]}
    if "embedding" in result:
        return result["embedding"]
    return result["data"][0]["embedding"]


def build_centroids(server_url, experts_path=EXPERTS, out_path=CENTROIDS):
    """Embed anchor prompts per expert, average into centroid. Saves JSON."""
    experts = json.loads(experts_path.read_text(encoding="utf-8"))
    centroids = {}
    for exp in experts["experts"]:
        vecs = []
        for prompt in exp["anchor_prompts"]:
            print(f"  embedding anchor for {exp['id']}: {prompt[:40]}...", flush=True)
            vecs.append(embed_via_server(server_url, prompt))
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        centroids[exp["id"]] = centroid
    out_path.write_text(json.dumps(
        {"centroids": centroids, "dim": dim,
         "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2) + "\n", encoding="utf-8")
    print(f"Centroids saved: {out_path}")
    return centroids


def route(server_url, query, experts_path=EXPERTS,
          centroids_path=CENTROIDS, top_k=1):
    """Embed query, return top-k experts by cosine similarity."""
    experts = json.loads(experts_path.read_text(encoding="utf-8"))
    if not centroids_path.exists():
        raise FileNotFoundError(
            f"Centroids not built. Run with --build-centroids first.")
    centroids = json.loads(centroids_path.read_text(encoding="utf-8"))["centroids"]
    qvec = embed_via_server(server_url, query)
    scores = []
    for exp in experts["experts"]:
        sim = cosine(qvec, centroids[exp["id"]])
        scores.append({"id": exp["id"], "model": exp["model"],
                       "similarity": round(sim, 4)})
    scores.sort(key=lambda s: s["similarity"], reverse=True)
    return scores[:top_k], scores


def run_expert(runtime, model, prompt, max_tokens=128, timeout=180):
    """Run a single expert via llama-cli. Returns output text."""
    cmd = [str(runtime), "-m", str(model), "-ngl", "0", "-t", "1",
           "-c", "512", "-n", str(max_tokens),
           "--single-turn", "--no-warmup", "-p", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    return result.stdout, result.returncode


def main():
    ap = argparse.ArgumentParser(description="Sparse MoE router")
    ap.add_argument("--server", default="http://localhost:8080",
                    help="llama-server URL with --embedding enabled")
    ap.add_argument("--build-centroids", action="store_true",
                    help="Build anchor centroids and exit")
    ap.add_argument("--query", type=str, help="Prompt to route")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--runtime", type=pathlib.Path,
                    help="llama-cli for running selected expert")
    ap.add_argument("--run", action="store_true",
                    help="Also run the selected expert on the query")
    args = ap.parse_args()

    if args.build_centroids:
        build_centroids(args.server)
        return 0
    if not args.query:
        ap.error("--query required (or --build-centroids)")
    selected, all_scores = route(args.server, args.query, top_k=args.top_k)
    print(json.dumps({"query": args.query[:80], "all_scores": all_scores,
                      "selected": selected}, indent=2))
    if args.run:
        if not args.runtime:
            ap.error("--runtime required with --run")
        for s in selected:
            model_path = ROOT / s["model"]
            if not model_path.is_file():
                print(f"Model not found: {model_path}", flush=True)
                continue
            print(f"\n--- Running expert {s['id']} ---", flush=True)
            out, rc = run_expert(args.runtime.resolve(), model_path.resolve(),
                                 args.query)
            print(out[-3000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
