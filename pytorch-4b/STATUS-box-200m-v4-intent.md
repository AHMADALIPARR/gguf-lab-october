<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# box-200m-v4-intent status (measured 2026-10-16)

Pipeline: `runs/box-200m-intent/ei.pt` → continue-train on corpus v4 (1200 steps) → intent SFT 600 → EI 1 → eval. Driver: `runs/box-200m-v4/pipeline.sh`; notes: `runs/box-200m-v4/PROGRESS.md`.

## Corpus v4 (`scripts/build_corpus_v4.py`, `pytorch-4b/data/v4/stats.json`, 32k tokenizer)
| source | docs | unique tokens | train tokens (after repeat) |
|---|---:|---:|---:|
| SNAPKITTYWEST/TWIN | 1635 | 3,517,220 | 3,444,179 |
| SNAPKITTYWEST/ai-free (master, ×2) | 221 | 971,173 | 1,914,176 |
| legacy-logic-archive (allowlist export, capped 4M) | 612 | 4,001,122 | 3,934,869 |
| total | 2468 | 8,489,515 | 9,293,224 (val 153,379) |

legacy-logic-archive: `native-recheck/bin/fetch-corpus.pl` fetched 9 snapshots. `normalize.pl` exited 1 (missing_git_blobs 757, git_blob_mismatches 0). `export-training.pl` exported 2627 allowlisted files (all otp-18.0, Apache-2.0/MIT, 46.2 MB). `report.pl` and `verify-native-recheck.pl` did not run because the native-recheck outputs and the historical toolchains are missing.

## Continue-train (`runs/box-200m-v4/ct.log`, batch 4 × accum 2, seq 256, lr 1e-4→1e-5, old-frac 0.25 from snapkitty2)
| step | train loss | new_val | old_val |
|---:|---:|---:|---:|
| 0 | n/a | 5.6236 | 3.9008 |
| 200 | 3.6350 | 4.1872 | 3.9148 |
| 400 | 3.1836 | 3.9203 | 3.9065 |
| 600 | 3.0728 | 3.7480 | 3.9136 |
| 800 | 3.0582 | 3.6610 | 3.8920 |
| 1000 | 3.0754 | 3.6043 | 3.8788 |
| 1200 | 2.9747 | 3.5769 | 3.8698 |
old_val here is `continue_train.loss_on` on `data/val.bin` (128 windows). It is not the agent.py `old_val_loss` (3.5383 for ei).

## Held-out zeros: cause and fix
- Cause: `agent.py rollout` capped the final answer at 24 tokens. worm_seal answers are 39–47 tokens (100/100 over the cap), and 64/100 viterbi answers are over it. Even with no cap, old ei.pt does not copy the `<result>`: it degenerates into repeats or wrong hex.
- Harness fix: final budget = max(24, len(result)+4). `DSS_FIXED_FINAL=1` restores the old cap.
- Data: `dss/data/ruby_train_targeted.jsonl`, 4000 viterbi + 4000 worm_seal from `dss-host answer`, with test prompts excluded (`kernels/ruby_ports/make_ruby_targeted.py`). SFT pool: train_aug + phi_train + ruby_train + targeted (train_v2 DSS bulk dropped).

## Held-out intent (`test_ruby_mixed_balanced`, n=1224)
| task | old ei (24-cap) | old ei (fixed) | v4 base | v4 sft | v4 ei1 (fixed) | v4 ei1 (24-cap) |
|---|---:|---:|---:|---:|---:|---:|
| next | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| card | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| isdss | 0.99 | 0.99 | 0.995 | 0.975 | 1.0 | 1.0 |
| bound | 0.29 | 0.29 | 0.23 | 0.41 | 0.48 | 0.48 |
| phi_normalize | 0.18 | 0.18 | 0.0 | 0.58 | 0.54 | 0.54 |
| phi_mul | 0.08 | 0.08 | 0.0 | 0.45 | 0.40 | 0.40 |
| phi_conj | 0.63 | 0.63 | 0.0 | 0.82 | 0.82 | 0.82 |
| phi_norm | 0.26 | 0.26 | 0.0 | 0.68 | 0.66 | 0.66 |
| viterbi | 0.0 | 0.0 | 0.0 | 0.77 | 0.97 | 0.42 |
| permission | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| governance | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| worm_seal | 0.0 | 0.0 | 0.0 | 0.13 | 0.27 | 0.0 |
| **all** | 0.5441 | 0.5441 | 0.3448 | 0.7377 | **0.7663** | 0.6993 |
agent.py old_val_loss: v4 base 3.4038, sft 3.5725, ei1 3.5536 (old ei 3.5383). EI kept 706 verified samples. Raw: `runs/box-200m-v4-intent/metrics.jsonl`, `runs/box-200m-v4/ei-old-newharness.json`, `ei-new-legacyharness.json`.

## Live host_eval (`box-200m-v4-intent-ei.gguf`, 804,398,048 B, sha256 d64bbf6f41db6a4af6f373879ef0ce7fff17f2920c622a2c33f9176d0c157b96)
| suite | n | v4 ei1 | old ei |
|---|---:|---:|---:|
| intent balanced | 1224 | 1.0000 | 1.0000 |
| intent OOD | 1202 | 0.9476 ± 0.0064 | 0.9958 ± 0.0019 |
OOD tasks below 1.0: phi_mul 0.67, worm_seal 0.82, phi_conj 0.92, phi_norm 0.97, permission 0.99. Old ei had only permission 0.98 and worm_seal 0.97. Live OOD regressed, so live intent-routing robustness went down. DSS/phi live suites were not re-run. Raw: `runs/box-200m-v4-intent/live/`.
