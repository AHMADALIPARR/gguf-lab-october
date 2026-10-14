# Phi kernel (sovereign-ruby-kernel) as a second checked tool

The source is SNAPKITTYWEST/sovereign-ruby-kernel, cloned at
`/workspace/sovereign-ruby-kernel`. Training on it was authorized by the
licensor, the trustee of its licenses. All numbers below are measured.

## 1. What the Ruby code actually does

The core files are `lib/ast/phi_algebra.rb`, `lib/ast/rewrite.rb` and
`lib/kernel/symbolic_kernel.rb`. `PhiAlgebra` builds ASTs from literals,
`phi(k)`, `phi_hat(k)`, `add` and `mul`, plus a Galois conjugate (swaps
phi and phi_hat) and a norm node (x times its conjugate). It **evaluates
everything in f64**, using `PHI = (1 + Math.sqrt(5)) / 2.0`. `Rewrite` has
four rules: `phi^2 -> phi + 1`, `1*x -> x`, `x + 0 -> x`, and
`norm -> x * conj(x)`.

**Spec run** (Ruby 3.3.8, installed for this work; `ruby spec/kernel_spec.rb`):
- As shipped: 24 tests, **12 pass and 12 fail**. All 12 failures are
  ``undefined method `iso8601' for Time`` (the WORM module calls
  `Time#iso8601` without `require 'time'`).
- With `ruby -rtime`: **24/24 pass**.
- `spec/agent_tests.rb` does not load (`cannot load such file --
  spec/agents_framework`).

### README claims

| claim | verdict | evidence |
|---|---|---|
| "Canonical TRS: 388.985128" | **Implemented (f64), value correct to the printed 6 decimals** | Computed exactly, TRS = (534 + 872 phi)/5 = 388.98512763798166152... Ruby's f64 gives 388.98512763798175. The printed value is this rounded to 6 decimals (difference 3.6e-7). The spec accepts any value within 0.01 and `verify!` within 0.001. |
| "Galois norm: TRS x sigma(TRS) in Q" | **True, but Asserted-only in the code** | Exact norm = -1916/5 = -383.2, which is rational. Ruby computes -383.2000000000001 in f64. The spec test "norm is rational" only checks that the value is a finite Float. |
| "Confluence: order-independent (proven by exhaustive check)" | **Asserted-only (not exhaustive)** | `confluent?` normalizes one term, the canonical tree, under two rule orders (forward and reversed) and compares the results. It covers no other terms, critical pairs or orderings. The repo has no exhaustive check. |
| "phi^2 = phi + 1 normalization" | **Partial** | Only the exponent-2 node is rewritten. On the canonical tree, `phi^3`..`phi^7` remain (4, 4, 4, 8 and 4 occurrences). `normalize(phi(4))` returns `(phi 4)`, so terms are not reduced to a + b phi. |

## 2. Exact Rust port (`dss/host/src/phi.rs`)

The port works in Z[phi] with `i128` and checked arithmetic. Overflow
returns an error, and there is no f32/f64 anywhere in the module. It uses:
- phi^k = F(k-1) + F(k) phi;
- (a + b phi)(c + d phi) = (ac + bd) + (ad + bc + bd) phi;
- conj(a + b phi) = (a + b) - b phi;
- norm = a^2 + ab - b^2.

Unit tests check phi^2 = 1 + phi, x * conj(x) = norm(x), and
norm(534 + 872 phi) = -1916 * 5.

**Cross-check against Ruby** (`kernels/phi/phi_crosscheck.py`, output in
`kernels/phi/crosscheck_results.txt`). Ruby's own `PhiAlgebra.evaluate`
(f64) is compared with the Rust exact result evaluated to 60 digits.
Agreement means a relative error of at most 1e-9 of the summed term
magnitudes. There are 500 random inputs per row:

| inputs | normalize | mul | conj | norm |
|---|---|---|---|---|
| small (abs(coef) <= 50, k <= 12) | 500/500 | 500/500 | 500/500 | 500/500 |
| large (abs(coef) <= 10^9, k <= 60) | 500/500 | 500/500 | 500/500 | 500/500 |

The maximum relative error is 2.1e-15. However, Ruby's f64 cannot return
the exact integer norm for large inputs. Rounding Ruby's norm gives the
exact integer in **500/500 small cases but only 5/500 large cases**, which
is why the kernel answers come from the exact Rust port.

## 3. Intents and host routing

There are four new intents: `PHI_NORMALIZE`, `PHI_MUL`, `PHI_CONJ` and
`PHI_NORM`. Strict templates (`dss/host/src/parse.rs`):

```
phi normalize 2 phi^5 - 1 phi^2 + 3 ->        8 + 9 phi
phi mul (2 + 3 phi) (-1 + 1 phi) ->           1 + 2 phi
phi conj (3 - 2 phi) ->                       1 + 2 phi
phi norm (534 + 872 phi) ->                   -9580
```

Parser rules:
- integers are signed and must have abs <= 10^12, with no leading zeros
  and no `+` sign;
- exponents are 0..80;
- a polynomial has 1..8 terms;
- `(3 + -2 phi)` is rejected.

The same `dss-host` binary routes all 8 intents across both kernels (DSS
and phi). It returns the kernel result verbatim, or `ERROR: ...`.

## 4. Data and model

- **Train:** 8000 phi prompts (`dss/data/phi_train.jsonl`, 2000 per task,
  abs(coef) <= 50 (25% up to 1000), k <= 12 (25% up to 20)), mixed with
  the 40,000 DSS intent rows (`train_v2` + `train_aug`).
- **Test:** a balanced phi test of 100 per task, with prompts disjoint from
  training, and an OOD phi test of 100 per task with abs(coef) between
  10^6 and 10^9 and k up to 60. Both are combined with the DSS tests in
  `test_mixed_balanced.jsonl` (900) and `test_mixed_ood.jsonl` (802).
- **Training:** started from `dss-intent-aug` `ei.pt`. SFT 500 steps with
  result tokens masked and 4 old-corpus replay windows per step, then 1
  expert-iteration round. 8 threads, 23:15-23:33 PT. Log:
  `kernels/phi/phi-intent-metrics.jsonl`.
- **Old-val loss:** 2.9818 -> 3.0845 after SFT -> **3.0754** after expert
  iteration (+0.094 over the start).
- **Export:** `runs/phi-intent/phi-intent.gguf` (not committed).
  - Token parity: 500/500 DSS texts, 31/32 mixed texts, and 800/800 phi
    prompt+trajectory texts.
  - Trajectory parity between PyTorch and llama.cpp through the host
    (`intent-gen`, 3 prompts per task across both test sets): **24/24
    identical**.

## 5. Live llama.cpp results through the Rust host (host returns the kernel answer)

| task | balanced n | intent correct | parse reject | acc | OOD n | intent correct | parse reject | acc ± SE |
|---|---|---|---|---|---|---|---|---|
| next | 100 | 1.00 | 0 | 1.00 | 100 | 1.00 | 0 | 1.00 |
| card | 100 | 1.00 | 0 | 1.00 | 100 | 1.00 | 0 | 1.00 |
| is-DSS | 200 | 1.00 | 0 | 1.00 | 101 | 1.00 | 0 | 1.00 |
| bound | 100 | 1.00 | 0 | 1.00 | 101 | 1.00 | 0 | 1.00 |
| phi_normalize | 100 | 1.00 | 0 | 1.00 | 100 | 1.00 | 0 | 1.00 |
| phi_mul | 100 | 1.00 | 0 | 1.00 | 100 | 1.00 | 0 | 1.00 |
| phi_conj | 100 | 1.00 | 0 | 1.00 | 100 | 1.00 | 0 | 1.00 |
| phi_norm | 100 | 1.00 | 0 | 1.00 | 100 | 0.99 | 0 | 0.99 ± 0.01 |
| **all** | 900 | 1.000 | 0 | **1.000** (0/900 wrong) | 802 | 0.999 | 0 | **0.999 ± 0.001** |

- **The one OOD miss** is `phi norm (-253445872 + 660947802 phi) ->`. The
  model emitted `[INTENT: PHI_NORMALIZE]`, so the host returned
  `ERROR: malformed or wrong-task call` rather than an answer.
- **DSS regression check:** the DSS rows are unchanged from round 6, at
  500/500 balanced and 402/402 OOD. Both live-failure prompts are still
  answered correctly.
- **Per kernel:**
  - DSS: balanced 500/500, OOD 402/402.
  - Phi: balanced 400/400, OOD 399/400 (0.9975 ± 0.0025).

Results: `dss/results/live/summary-phi.jsonl`.
