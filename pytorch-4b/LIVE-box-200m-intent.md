<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# box-200m-intent: live host evals (measured 2026-10-15, 16:49–17:41 PT)

GGUF: `box-200m-intent-ei.gguf` (804,398,048 B, sha256 e755c13a…398f). Command: `dss/host_eval.py intent <gguf> <test> <prefix>` (Rust `dss-host eval` + offline `grade.py`). Raw outputs: `live/box200m-ei-*.{json,out,tsv,prompts.txt}`.

| suite | test file | n | all acc | ± SE | non-1.0 tasks |
|---|---|---:|---:|---:|---|
| intent balanced | test_ruby_mixed_balanced | 1224 | 1.0000 | 0.0000 | none |
| intent OOD | test_ruby_mixed_ood | 1202 | 0.9958 | 0.0019 | permission 0.98, worm_seal 0.97 |
| DSS/phi balanced | test_mixed_balanced | 900 | 1.0000 | 0.0000 | none |
| DSS/phi OOD | test_mixed_ood | 802 | 1.0000 | 0.0000 | none |

The reference `ruby-intent` run scored 1.0000 balanced and 0.9983 OOD on the same intent suites.

These numbers are much higher than the heldout `agent.py eval` accuracy (0.5441), and the two do not measure the same thing. In live intent mode the model only picks the tool and its arguments, and the host computes the answer. The heldout eval grades the model's own generated answer. `dpo.pt` was exported (`box-200m-intent-dpo.gguf`, sha256 60f6d586…b1ef), but it has not been live-evaluated.
