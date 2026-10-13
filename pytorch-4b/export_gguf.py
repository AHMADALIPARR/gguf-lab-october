#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Export a Transformer4B checkpoint (model.py) to a llama-arch GGUF.

RoPE note: model.py rotates adjacent pairs (view_as_complex on (..., D/2, 2)),
i.e. the original Meta/"interleaved" convention, which is exactly what
llama.cpp's LLAMA arch (rope type NORM) does. So q/k are written as-is; no
HF-style permute is applied.

Usage:
  python export_gguf.py CKPT.pt OUT.gguf [--config small] [--tokenizer tokenizer/tokenizer.json] [--dtype f32|f16]
CKPT may be a raw state_dict or a dict with "model"/"model_state_dict"/"state_dict"
and optionally "config" (dict of Config fields).
"""
import argparse, json, os, sys
import numpy as np
import torch
import gguf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import CONFIGS, config_to_dict  # noqa: E402


def load_ckpt(path, config_name):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, cfgd = ck, None
    if isinstance(ck, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]; break
        c = ck.get("config") if isinstance(ck.get("config"), dict) else ck.get("cfg")
        if isinstance(c, dict):
            cfgd = c
    if cfgd is None:
        cfgd = config_to_dict(CONFIGS[config_name]())
    sd = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in sd.items()}
    return sd, cfgd


def add_tokenizer(w, tok_path, vocab_size):
    t = json.load(open(tok_path, encoding="utf-8"))
    m = t["model"]
    assert m["type"] == "BPE", m["type"]
    vocab = dict(m["vocab"])
    for a in t.get("added_tokens", []):
        vocab[a["content"]] = a["id"]
    n = max(vocab_size, max(vocab.values()) + 1)
    toks = [f"[PAD{i}]" for i in range(n)]
    types = [gguf.TokenType.UNUSED] * n
    for s, i in vocab.items():
        toks[i] = s; types[i] = gguf.TokenType.NORMAL
    specials = {}
    for a in t.get("added_tokens", []):
        types[a["id"]] = gguf.TokenType.CONTROL if a.get("special") else gguf.TokenType.USER_DEFINED
        specials[a["content"]] = a["id"]
    merges = [" ".join(x) if isinstance(x, list) else x for x in m["merges"]]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    if "<s>" in specials: w.add_bos_token_id(specials["<s>"])
    if "</s>" in specials: w.add_eos_token_id(specials["</s>"])
    if "<unk>" in specials: w.add_unk_token_id(specials["<unk>"])
    if "<pad>" in specials: w.add_pad_token_id(specials["<pad>"])
    w.add_add_bos_token(False)


def export(sd, cfg, tok_path, out, dtype="f32", name="transformer4b"):
    H, nh, nkv = cfg["hidden_dim"], cfg["n_heads"], cfg["n_kv_heads"]
    L, V = cfg["n_layers"], cfg["vocab_size"]
    w = gguf.GGUFWriter(out, "llama")
    w.add_name(name)
    w.add_context_length(cfg["max_seq_len"])
    w.add_embedding_length(H)
    w.add_block_count(L)
    w.add_feed_forward_length(cfg["ffn_dim"])
    w.add_head_count(nh)
    w.add_head_count_kv(nkv)
    w.add_rope_dimension_count(H // nh)
    w.add_rope_freq_base(float(cfg["rope_theta"]))
    w.add_layer_norm_rms_eps(float(cfg["norm_eps"]))
    w.add_vocab_size(V)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if dtype == "f16" else gguf.LlamaFileType.ALL_F32)
    add_tokenizer(w, tok_path, V)

    def put(gname, key):
        a = sd[key].detach().float().cpu().numpy()
        if dtype == "f16" and a.ndim == 2:
            a = a.astype(np.float16)
        w.add_tensor(gname, np.ascontiguousarray(a))

    put("token_embd.weight", "tok_emb.weight")
    put("output_norm.weight", "norm.weight")
    put("output.weight", "head.weight" if "head.weight" in sd else "tok_emb.weight")
    for i in range(L):
        p = f"layers.{i}."
        for g, k in [("attn_norm", "attn_norm.weight"), ("attn_q", "attn.wq.weight"),
                     ("attn_k", "attn.wk.weight"), ("attn_v", "attn.wv.weight"),
                     ("attn_output", "attn.wo.weight"), ("ffn_norm", "ffn_norm.weight"),
                     ("ffn_gate", "ffn.gate.weight"), ("ffn_up", "ffn.up.weight"),
                     ("ffn_down", "ffn.down.weight")]:
            put(f"blk.{i}.{g}.weight", p + k)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--config", default="small", choices=list(CONFIGS))
    ap.add_argument("--tokenizer", default=os.path.join(HERE, "tokenizer", "tokenizer.json"))
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"])
    a = ap.parse_args()
    sd, cfg = load_ckpt(a.ckpt, a.config)
    export(sd, cfg, a.tokenizer, a.out, a.dtype)
    print(f"wrote {a.out}  cfg={cfg}")


if __name__ == "__main__":
    main()
