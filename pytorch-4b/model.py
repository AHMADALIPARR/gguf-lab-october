#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""4B dense transformer recreated from the unified tensor manifest pattern.

Architecture follows the observed patterns in manifests/unified.jsonl:
- GQA attention (mellum: 32 Q heads, 4 KV heads; qwen: similar ratio)
- SwiGLU FFN (gate/up/down projections)
- RMSNorm (not LayerNorm)
- RoPE positional encoding
- Dense FFN (not MoE — the 4B target doesn't need expert routing)

Target: ~4B parameters. Exact count printed at build time.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 32000
    hidden_dim = 3072
    n_layers = 36
    n_heads = 24
    n_kv_heads = 8
    ffn_dim = 8192
    max_seq_len = 4096
    rope_theta = 500000.0
    norm_eps = 1e-6
    dropout = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def precompute_rope(dim, seq_len, theta=500000.0):
    """RoPE frequencies. dim = head_dim."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rope(xq, xk, freqs):
    """xq, xk: (B, T, heads, head_dim). freqs: (T, head_dim//2) complex."""
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.hidden_dim // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.hidden_dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.hidden_dim, bias=False)

    def forward(self, x, freqs, mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs)

        # Repeat KV heads for GQA
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x, freqs, mask=None):
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer4B(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.register_buffer(
            "rope_freqs",
            precompute_rope(cfg.hidden_dim // cfg.n_heads,
                            cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

    def forward(self, ids, mask=None):
        B, T = ids.shape
        x = self.tok_emb(ids)
        freqs = self.rope_freqs[:T]
        for layer in self.layers:
            x = layer(x, freqs, mask)
        x = self.norm(x)
        return self.head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = Config()
    model = Transformer4B(cfg)
    n = model.param_count()
    print(f"Parameters: {n:,} ({n/1e9:.2f}B)")
    print(f"Layers: {cfg.n_layers}, hidden: {cfg.hidden_dim}, "
          f"heads: {cfg.n_heads}/{cfg.n_kv_heads}, ffn: {cfg.ffn_dim}")

    # Shape check
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        out = model(ids)
        assert out.shape == (2, 16, cfg.vocab_size), f"bad shape {out.shape}"
        print(f"Forward OK: {tuple(out.shape)}")
