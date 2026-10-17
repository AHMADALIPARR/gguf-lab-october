<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# box-200m-v4 progress (resume notes)
- Corpus: `scripts/build_corpus_v4.py` -> `pytorch-4b/data/v4` (log `runs/build_corpus_v4.log`).
- Continue-train: PID 1145069, `runs/box-200m-v4/ct.log` (accum 2, 1200 steps). An earlier accum-4 attempt was aborted at step 50 (`aborted-accum4/`).
- Zero cause found: `agent.py rollout` capped final answer at 24 tokens; worm_seal answers 39-47 tok (100/100 over), viterbi 64/100 over. Harness fix: budget = max(24, len(result)+4); `DSS_FIXED_FINAL=1` restores legacy. Targeted data: `dss/data/ruby_train_targeted.jsonl` (4000 viterbi + 4000 worm_seal, dss-host answers, test prompts excluded) via `kernels/ruby_ports/make_ruby_targeted.py`.
- Remaining steps chained in `runs/box-200m-v4/pipeline.sh` (resumable; rerun with nohup). Log: `pipeline.log`.
- 2026-10-16 15:58 PDT continue_train done (final.pt)
- 2026-10-16 15:58 PDT intent SFT/EI started
- 2026-10-16 18:29 PDT intent SFT/EI done
- 2026-10-16 18:29 PDT EI1 heldout all 0.7663 (viterbi 0.97, worm_seal 0.27); old-ei rescore running
- 2026-10-16 18:49 PDT old ei.pt heldout (new harness) done
- 2026-10-16 18:49 PDT old ei.pt w/ budget fix: all 0.5441, viterbi 0.0, worm_seal 0.0 (unchanged)
- 2026-10-16 19:02 PDT new ei.pt heldout (legacy 24-tok harness) done
- 2026-10-16 19:02 PDT gguf exported
- 2026-10-16 19:09 PDT live balanced done
- 2026-10-16 19:15 PDT live ood done
- 2026-10-16 19:15 PDT pipeline complete
- 2026-10-16 19:16 PDT STATUS.md + HANDOFF.md written; live bal 1.0, ood 0.9476
