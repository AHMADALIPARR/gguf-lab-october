<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# HANDOFF: box-trained 200.8M model → GPU

Status: pretrain `runs/box-200m-b/final.pt` is done. Intent SFT→EI→DPO on that checkpoint is done (`runs/box-200m-intent/`); see `STATUS.md` there (pushed copy: `pytorch-4b/STATUS-box-200m-intent.md`) for measured heldout numbers. Losses below are only the ones already measured.

## Intent stage (measured 2026-10-15)
- Path: `box-200m-b/final.pt` → SFT 600 → EI 1 → DPO 120 (`penalty_dpo` / pairs from `neg-rl/pairs.jsonl`) → heldout `agent.py eval … --mode intent`.
- Heldout all-acc: base 0.0000 → sft 0.5106 → ei1 0.5441 → dpo 0.5458. Old-val: 3.0202 → 3.5954 → 3.5383 → 3.5388.
- DPO raised pair pref_acc 0.5532 → 0.8723; heldout task accuracy stayed flat.
- Still zero: `worm_seal`, `viterbi`. Weak: `phi_mul` 0.08, `phi_normalize` 0.18, `bound` 0.30, `phi_norm` 0.27. Strong: `next`/`card`/`permission`/`governance` 1.0, `isdss` 0.99.

- Live host eval of `runs/box-200m-intent/box-200m-intent-ei.gguf` (804,398,048 B), `dss/host_eval.py intent`. Intent balanced 1.0000 (n=1224). Intent OOD 0.9958 ± 0.0019 (n=1202; permission 0.98, worm_seal 0.97). DSS/phi balanced 1.0000 (n=900). DSS/phi OOD 1.0000 (n=802). In live mode the host executes the tool the model chose, so these numbers are not comparable to the heldout generation accuracy above. Details: `runs/box-200m-intent/LIVE.md` (pushed copy: `pytorch-4b/LIVE-box-200m-intent.md`), status `runs/box-200m-intent/STATUS.md` (pushed copy: `pytorch-4b/STATUS-box-200m-intent.md`).
- Primary deliverable: the EI GGUF, because DPO left heldout flat. `box-200m-intent-dpo.gguf` (same size) is exported for completeness; a focused OOD smoke (n=220) on 2026-10-16 matched EI on permission 0.98 and worm_seal 0.97 (see LIVE.md). Full four-suite live eval still not run.

## Base pretrain eval (`runs/box-200m-b.log`, measured)

| step | train loss | new_val | old_val |
|---:|---:|---:|---:|
| 0 | n/a | 4.9701 | 5.0944 |
| 500 | 3.5546 | 4.5222 | 4.4253 |
| 1000 | 3.4618 | 4.1134 | 3.9176 |
| 1500 | 3.3140 | 3.9687 | 3.7234 |

## v4 stage (measured 2026-10-16) — see `runs/box-200m-v4-intent/STATUS.md` (copy `pytorch-4b/STATUS-box-200m-v4-intent.md`)
- Corpus v4 (TWIN 3.52M, ai-free 0.97M ×2, legacy-logic-archive 4.00M capped; 9.29M train tokens). Continue-trained `box-200m-intent/ei.pt` 1200 steps: new_val 5.6236→3.5769, old_val 3.9008→3.8698.
- Held-out zeros: the cause was the 24-token final-answer cap in `agent.py rollout` plus weak copying. The cap is fixed (`DSS_FIXED_FINAL=1` restores the old one). Targeted viterbi/worm_seal SFT data was added. Old ei.pt under the fixed harness is unchanged at 0.5441 (viterbi 0, worm_seal 0).
- v4 SFT→EI1 held-out all 0.7663 (viterbi 0.97, worm_seal 0.27; under the old 24-token cap: 0.6993, viterbi 0.42, worm_seal 0.0).
- Live: `runs/box-200m-v4-intent/box-200m-v4-intent-ei.gguf` (sha256 d64bbf6f…7b96), intent balanced 1.0000, OOD 0.9476 (old ei 0.9958). This is an OOD regression (phi_mul 0.67, worm_seal 0.82). Pick v4 for held-out generation and old ei for live OOD routing.
- legacy-logic-archive: normalize exited 1 (757 missing git blobs). report/verify need the native-recheck toolchains and were not run.

## What was trained on the box
- Architecture: `model.py` `Transformer4B` (GQA, SwiGLU, RMSNorm, RoPE, untied embeddings), 32k tokenizer `tokenizer/tokenizer.json`.
- Config: hidden 1024, layers 12, heads 16, kv_heads 4, ffn 2816, max_seq_len 256, rope_theta 10000, vocab 32000 → 200.8M params.
- Size probes (`scripts/size_probe.py`, AdamW fp32, seq 256, 8 threads; random tokens, 6 to 8 steps, median step):

| params | h/L/heads/kv/ffn | batch | peak RSS | tok/s |
|---|---|---|---|---|
| 112.1M | 768/10/12/4/2048 | 4 | 3.40 GB | 873 |
| 112.1M | same | 16 | 7.52 GB | 1053 |
| 157.8M | 896/12/14/2/2432 | 4 | 4.37 GB | 644 |
| 200.8M | 1024/12/16/4/2816 | 4 | 5.46 GB | 555 |
| 200.8M | same | 8 | 7.39 GB | 518 |
| 271.9M | 1152/14/18/6/3072 | 4 | 6.97 GB | 276 |
| 357.2M | 1280/16/20/4/3456 | 4 | 8.95 GB | 292 |

  About 10 GB was actually free on the 15 GB box. 357M left about 1 GB of headroom. The 272M and 357M models trained at under 300 tok/s, which is about 1M tokens per hour. 200.8M was the largest model that kept more than 3 GB free and still trained at over 500 tok/s.
- Data: `data/snapkitty2/tokens.bin` (15.68M train tokens, built by `scripts/build_corpus_snapkitty.py`). Its val split is `data/snapkitty2/val.bin` (434k tokens, a 2% doc-level holdout). Each batch also takes 25% from the old corpus `data/tokens.bin`. Old-val is `data/val.bin`.
- Script: `scripts/continue_train.py --init 1024,12,16,4,2816 --batch 4 --accum 4 --seq 256 --lr 3e-4 --min-lr 3e-5 --warmup 200 --steps 6000`, AdamW (0.9, 0.95), wd 0.1, clip 1.0, cosine schedule.
- Intents: after pretraining, the four Ruby-kernel intents plus DSS/phi get the same intent SFT/EI stage as `runs/ruby-intent` (`dss/agent.py`). They are then evaluated live with `dss/host_eval.py intent`.

## Checkpoint format
`{"model": state_dict, "config": {hidden_dim, n_layers, n_heads, n_kv_heads, ffn_dim, max_seq_len, rope_theta, norm_eps, vocab_size, dropout}}`. Load it with `eval.load_model(path)`. The optimizer state is not saved, so resuming starts a new AdamW.

## Resuming on GPU
1. The same 200.8M config continues directly: load the state_dict, move it to cuda, and switch to bf16 autocast. You can raise `max_seq_len`, because RoPE has no learned position table. Fine-tune at the longer length.
2. The 3.82B config (`Config` in `model.py`: 3072/36/24/8/8192, seq 4096, rope_theta 5e5) has different shapes. These weights cannot be loaded into it. Use the box run as a pipeline, data and tokenizer check, then train 3.82B from scratch with `train.py --config 4b` on the same token files. Weights could only carry over with a growth method (e.g. width/depth expansion), and nothing like that was tested here.
3. Tokenizer and data are shared, so the uint16 `.bin` files can be reused as they are.
