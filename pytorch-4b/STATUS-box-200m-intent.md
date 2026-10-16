<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# box-200m-intent status (measured)

Pipeline: `box-200m-b/final.pt` → SFT 600 → EI 1 → DPO 120 → heldout eval. Done.

| stage | all acc | old_val | notes |
|---|---:|---:|---|
| base | 0.0000 | 3.0202 | pre-intent |
| sft | 0.5106 | 3.5954 | tools perfect; phi/worm/viterbi weak |
| ei1 | 0.5441 | 3.5383 | +3.4pp; worm_seal 0, viterbi 0 |
| dpo | 0.5458 | 3.5388 | pair pref_acc 0.55→0.87; heldout flat |

Still near zero after DPO: `worm_seal` 0.0, `viterbi` 0.0. Weak: `phi_mul` 0.08, `phi_normalize` 0.18, `bound` 0.30, `phi_norm` 0.27. Strong: `next`/`card`/`permission`/`governance` 1.0, `isdss` 0.99, `phi_conj` 0.63.

Artifacts: `sft.pt`, `ei.pt`, `dpo.pt`, `box-200m-intent-ei.gguf`, `dpo-heldout.json`.
