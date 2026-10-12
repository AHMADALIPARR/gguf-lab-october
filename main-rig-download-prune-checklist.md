# Main Rig Download + Prune Checklist
**October 2026 GGUF Lab** · generated 2026-10-12

Run on the NUC fleet / main rig. Staging VM cannot hold or process these.

## 0. Prerequisites
- `huggingface-cli` (or `hf`) from `pip install huggingface_hub`
- llama.cpp (recent build with Mellum / Qwen3.8 / ternary support)
- PyTorch + the pruning scripts from `SNAPKITTYWEST/sovereign-mimo-4b/prune/`
- Enough disk: ~50 GB for the five priority models + intermediates
- For 27B dense: ≥32 GB VRAM preferred; CPU offload possible but slow

## 1. Pull (corrected)
```bash
export GGUF_STORE=$HOME/gguf-lab
./pull.sh embed          # embeddinggemma-2-UD-Q4_K_XL.gguf  (~176 MB)
./pull.sh mellum         # Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf  (~8.1 GB)
./pull.sh qwen27         # Qwen3.8-27B-UD-Q4_K_M.gguf  (~16.5 GB)
./pull.sh bonsai         # Ternary-Bonsai-2-27B-PTQ1_0.gguf  (~5.9 GB)
./pull.sh xing           # xing4_0-29b-IQ4_NL.gguf  (~20.1 GB)
# Flash Next 177B is offload-only — pull manually if needed
```

Verified live filenames (2026-10-12):
- EmbeddingGemma: `embeddinggemma-2-UD-Q4_K_XL.gguf` (old name in script was wrong)
- Mellum 2.1: `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`
- Qwen3.8-27B: `Qwen3.8-27B-UD-Q4_K_M.gguf`
- Ternary Bonsai 2: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
- Xing 4.0: `xing4_0-29b-IQ4_NL.gguf`

## 2. Lightweight validation (fits in low RAM)
```bash
# Confirm GGUF magic + basic metadata without loading weights
python3 -c "
import struct, sys
p=sys.argv[1]
with open(p,'rb') as f:
    assert f.read(4)==b'GGUF'
    v=struct.unpack('<I', f.read(4))[0]
    tc,kc=struct.unpack('<QQ', f.read(16))
    print(f'{p}: version={v} tensors={tc} kvs={kc}')
" path/to/model.gguf
```

## 3. Prune order (from prune/plan.md)
1. **Mellum 2.1** (first real target)  
   - Expert-first: drop lowest-utility experts (router-weight analysis), then structured width prune on shared blocks.  
   - Watch for routing collapse. Validate coding/tool-use benchmarks after each cut.

2. **Qwen3.8-27B** (main workhorse)  
   - Dense: layer-drop by attention variance → hidden-width projection → head prune → MLP shrink.  
   - Target: ~40% params to fit 11B NUC, or comfortable main-rig footprint.  
   - Agentic gate: run ablation harness first (13 global layers, Q-head-output granularity, never KV heads). Prune around localized circuits.

3. **Xing 4.0** (MoE)  
   - Same expert-first approach as Mellum. 4B active already sparse — prune experts, not width.

4. **Ternary Bonsai 2**  
   - **Do not conventionally prune or requantize.** Already ~1.58 bits/weight. Study ternary sensitivity per layer only.

5. **Flash Next 177B**  
   - Offload-only research target. No local prune run.

## 4. Compression after prune (from compress/plan.md)
- Always re-quantize from the pruned BF16/F16, never quantize-then-prune.
- Mellum: IQ4_NL or Q3_K_M for NUC after prune.
- Qwen3.8-27B: IQ quants post-prune.
- EmbeddingGemma: leave at UD-Q4_K_XL (or higher).
- Xing: keep IQ4_NL.
- Bonsai: leave as PTQ1_0.
- Tool: `llama-quantize` from llama.cpp.

## 5. Suggested commands (example for Mellum)
```bash
# After pruning in PyTorch → export BF16/F16 GGUF
# Then:
llama-quantize pruned-mellum-f16.gguf mellum-pruned-IQ4_NL.gguf IQ4_NL
# Validate
llama-perplexity -m mellum-pruned-IQ4_NL.gguf -f wikitext-2-raw-v1
# + coding / tool-use suite
```

## 6. Notes
- Prefer Unsloth UD quants where both UD and plain exist.
- Ternary structure is destroyed by a second quantizer.
- Embedding quality is sensitive — do not aggressively quantize further.
- All models verified present on Hugging Face as of 2026-10-12.
