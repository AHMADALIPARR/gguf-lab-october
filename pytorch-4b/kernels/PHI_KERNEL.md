# Phi kernel (sovereign-ruby-kernel) as a second checked tool

The source is SNAPKITTYWEST/sovereign-ruby-kernel, cloned at
`/workspace/sovereign-ruby-kernel` (commit `52ff2a0`). The licensor, who
is the trustee of its licenses, authorized training on it. Its
`CLONE_GATE.md` lists "No AI training extraction without explicit written
authorization" as a term. All numbers below are measured. Every finding
names the file it comes from and applies only to that file.

## 1. Scope

**Clone-gate caveat.** `CLONE_GATE.md` says the repository is protected
by clone-gate verification. The public copy examined here may therefore
be partial. A component missing here may exist in the gated copy. Some
documents point to files this copy does not contain:
- `LICENSE-GPL` refers to a `LICENSE-APACHE`;
- `ruby-family-stack/README.md` refers to a `conformance/` directory,
  which is marked "planned".

**Read.** All 49 tracked files (14,557 lines):
- every file under `lib/` (ast, kernel, stages, swarm, golog);
- `bin/orchestrate`;
- `spec/`;
- `quantum-holographic/` (README, the `.pl` file, the XSL and the three
  XML submissions);
- all eight documents under `ruby-family-stack/`;
- `README.md`, `SWARM_BLUEPRINT.md`, `CLONE_GATE.md` and the licence
  files.

The large GOLOG sources were read by section and predicate structure,
and their test drivers were read in full.

**Executed** (box tools: Ruby 3.3.8, SWI-Prolog 9.2.9, Chez Scheme 10.0,
Guile 3.0, Python 3, xmllint and xsltproc; logs in `/tmp/rkrun`, not
committed). Results are in section 3.

**Not executed:**
- `dyna_golog_viterbi.ecl` (ECLiPSe) and `dyna_golog_viterbi.oz`
  (Mozart/Oz). Neither runtime is packaged for this Debian box, and
  neither was built from source.
- The Clojure branch of `stages/clojure_trs.rb` and the proof-file branch
  of `stages/axiom_proof.rb`. Their inputs (a Clojure project, a proof
  file) are not in this copy.
- The `ruby-family-stack/` documents describe 8 runtimes. Only CRuby
  3.3.8 was available, and I checked two numeric invariants against it.

**Used for the trained tool.** Only the phi arithmetic in
`lib/ast/phi_algebra.rb` (normalize, mul, conj, norm) was ported and
put behind the model (sections 5-8). No other component of the repo was
ported, trained on, or judged as a kernel here.

## 2. Inventory

Paths are relative to the repo root. "Checkable" means the file computes
something that has an exact or decidable answer an independent
implementation could verify.

### 2.1 Core Ruby kernel

| file | what it does | checkable |
|---|---|---|
| `lib/ast/node.rb` | Immutable AST node: `transform`, `fold`, `depth`, `to_sexp`, structural equality | yes (tree ops) |
| `lib/ast/phi_algebra.rb` | Builds terms (literal, phi^k, phi_hat^k, add, mul, Galois conjugate, norm) and evaluates them in f64; `trs_tree` builds the canonical sum | **yes (ported)** |
| `lib/ast/rewrite.rb` | Four rules (phi^2 -> phi+1, 1*x -> x, x+0 -> x, norm -> x*conj x); `normalize` to fixpoint with trace; `confluent?` compares two rule orders on one term | yes |
| `lib/kernel/symbolic_kernel.rb` | Canonical tree from `DEPTHS`/`BIASES` (ME, AN, KI, DINGIR); `evaluate`, conjugate, norm (f64); `verify!` checks the value against `CANONICAL_TRS = 388.985128` within 0.001 and drives the gate | yes |
| `lib/kernel/worm.rb` | In-process append-only SHA-256 hash chain (`seal`, `valid?`, `verify_entry`) | yes |
| `lib/kernel/governance.rb` | State machine init -> {verified, rejected}, verified -> {sealed, rejected}, sealed -> none, rejected -> init; each transition is WORM-sealed | yes (finite automaton) |
| `lib/stages/clojure_trs.rb` | Stage 1: runs `clojure -M -m sovereign.core` if a directory is given, else the inline Ruby evaluate/conjugate/norm | inline branch only |
| `lib/stages/apl_verify.rb` | Stage 2: sums the per-symbol subtrees (same `phi_algebra` f64 evaluator) and compares with `kernel.evaluate` within 1e-4 | yes |
| `lib/stages/axiom_proof.rb` | Stage 3: with a proof file, `verified` = no occurrence of `sorry`; otherwise `verified` = `Rewrite.confluent?` on the kernel tree | yes |
| `lib/stages/worm_seal.rb` | Stage 4: final WORM seal payload and the gate -> sealed transition | yes |
| `lib/swarm/agent.rb` | 8 agent roles with inbox/outbox queues, dispatching to the AST, Rewrite, WORM and Self-emission functions | partly |
| `lib/swarm/topology.rb` | 11-agent DAG (`sovereign_pipeline`), `tick` routing, DOT export | yes (graph) |
| `lib/swarm/self_synthesis.rb` | Emits Self-language prototype text from the AST forest | text emitter |
| `bin/orchestrate` | Runs stages 1-4, Self synthesis and topology | - |
| `spec/kernel_spec.rb` | 24 tests over the above | - |

### 2.2 `lib/golog/`

| file | language | what it does | checkable |
|---|---|---|---|
| `agents_framework.rb` | Ruby | `GologAgent` claim -> proof -> decision -> authorization -> execution state machine, hashed records, acyclic-authorization check, invariants | yes |
| `agent_dsl.rb` | Ruby | `AgentDSL` and `AgentCoordinator` over the framework, consensus threshold 0.67 | yes |
| `fsl_compiler.rb` | Ruby | FSL tokenizer, recursive-descent parser, compiler to "Opal" IR, forward-chaining `ConstraintEvaluator` | yes (parser) |
| `protocol_enforcement.rb` | Ruby | `ProtocolEnforcer`: decision register/prove/authorize/execute, cycle check, hash-chained audit trail | yes |
| `dyna_golog_viterbi.pl` | SWI-Prolog | Viterbi semiring (max, x), memoized HMM decoding (toy weather, POS tagger, CpG island), log-domain variant, GOLOG layer, axiom check and tests | **yes** |
| `dyna_golog_viterbi.ecl` | ECLiPSe | Same design with lib(ic)/store memo | yes (not run) |
| `dyna_golog_viterbi.oz` | Oz | Same design, OOP + store memo, test procedures | yes (not run) |
| `dyna_golog_viterbi_v1.py` | Python | v1.0 reference: semiring, memo, HMMs, lattice decode, GOLOG observe-decode, tests | **yes** |
| `holographic_golog_engine.py` | Python | Situation-calculus GOLOG interpreter (Prim, Seq, Choice, Star, Test, If, While, Conc, Prio, Proc) over Blocks, Elevator and Coffee domains | **yes** |
| `holographic_golog.scm` | Scheme | microKanren relational situation calculus (`holdingo`, Poss, `transo`/`finalo`/`executo`) | yes (does not load, see 3) |
| `constrained_golog.pl` | Prolog | Hand-written FD solver subset, lightweight linear store, constrained fluents (energy, time), `do_bounded/4`, Blocks/Elevator/Coffee demos | yes |
| `edaulc_agent.pl` | SWI-Prolog | Default-deny permission policy (`execution_decision/3`), content-class prohibitions, role -> competency map, KB self-check, 17 plunit tests | **yes** |
| `spec/agent_tests.rb` | Ruby | 53 minitest cases for the four Ruby files above | - |

### 2.3 `quantum-holographic/`

| file | what it does | checkable |
|---|---|---|
| `sgmt-crystallization.pl` | Declares 3 agents, EGG domains, constraints C1-C5 (`verify_constraint/2`) and an 8192-EGG crystallization driver | yes in form |
| `submissions/agent-{1,2,3}-*.xml` | Agent submissions with embedded Prolog/Datalog/CLP(FD) rule text | XML well-formedness |
| `xslt-normalize.xsl` | Normalizes a submission into `sgmt:crystallization` canonical XML | yes |
| `README.md` | Pipeline description and constraint list C1-C8 | - |

### 2.4 `ruby-family-stack/` (specification documents, no code)

| file | content |
|---|---|
| `ARCHITECTURE.md`, `README.md` | Five layers (spec, parsing, IR, backends, conformance); 8 runtime profiles; conformance levels 1-3 |
| `spec/LANGUAGE_SPECIFICATION.md` | Object model, dispatch, blocks/procs, strings/symbols, collections, exceptions, reflection, numerics |
| `spec/FORMAL_INVARIANTS.md` | Invariants I1-I28 (identity, MRO, closures, exceptions, scoping, numerics, reflection, typing, threads/fibers) |
| `parser/PARSER_ARCHITECTURE.md` | Lexer token types, simplified BNF, AST node types, normalization, symbol resolution, error recovery |
| `ir/INTERMEDIATE_REPRESENTATION.md` | IR node set, CFG, factorial example, optional SSA, IR invariants IR1-IR5 |
| `runtimes/RUNTIME_ABSTRACTION.md` | Capability interface and per-backend notes (CRuby, JRuby, Rubinius, mruby, TruffleRuby, Crystal, Opal, Mirah) |
| `governance/GOVERNANCE.md` | Specification authority, exception process (document, update spec, reject at parse time, conformance test), test format |

### 2.5 Other top-level files

- `README.md` and `SWARM_BLUEPRINT.md`: descriptions of the above.
- `CLONE_GATE.md`: licence and usage terms.
- `LICENSE-AGPL-V3`, `LICENSE-GPL`.
- `.gitignore`.

## 3. What ran, file by file

All of the following was run on copies of the repo. Nothing in the repo
was modified.

| file / command | result |
|---|---|
| `spec/kernel_spec.rb` | As shipped: 12/24 pass. All 12 failures are ``undefined method `iso8601' for Time`` from `lib/kernel/worm.rb`, which uses `Time#iso8601` without `require 'time'`. With `ruby -rtime`: **24/24 pass**. |
| `bin/orchestrate` | As shipped: stops with the same `iso8601` error. With `-rtime`: completes. It prints TRS = 388.985128, Norm = -383.2, the APL match, AXIOM verified = true, a valid chain and state = sealed. |
| `spec/agent_tests.rb` | As shipped it does not load: it uses `require_relative 'agents_framework'` from `spec/`, but the file is in `lib/golog/`. With symlinks added in a scratch copy: 53 runs, 0 failures, **43 errors**, so 10 pass. The errors are: 24 `ThreadError: deadlock; recursive locking` (`agents_framework.rb`: `claim` holds the mutex at line 84 and calls `transition_to`, which locks it again at line 67); 7 `ParseError: Unexpected token: eof` (`fsl_compiler.rb`); 6 `ArgumentError: Missing backing claims`; 4 `assert_not_nil` (not a minitest method); 2 references to undefined error classes. |
| `lib/golog/dyna_golog_viterbi.pl` | Runs. Semiring axioms pass, and the 4 tests print **ALL TESTS PASSED**. The printed decoded paths begin with an unbound variable (e.g. `[_12142,rainy,...]`). Toy-weather test: weight 0.015552. |
| `lib/golog/dyna_golog_viterbi_v1.py` | 5 tests pass, including toy weather 0.015552 (the same value as the Prolog file). `test_lattice_decode` then fails. Its own lattice has edges 0->1 (0.6), 1->3 (0.7), 0->2 (0.4), 2->3 (0.9). The max-product path is 0-1-3 with 0.42, but the test asserts path [0,2,3] with 0.36. The test's expectation is wrong for max-product, and the remaining tests do not run. |
| `lib/golog/holographic_golog_engine.py` | 5 tests pass. Then `test_star_and_test` fails: `assert dom.holds("at", final) == 3` after `run_online(star(...))`. |
| `lib/golog/holographic_golog.scm` | Does not load under Chez (unexpected EOF in the list at line 132) or Guile. The `holdingo` definition starting at line 132 is missing one closing parenthesis (the file ends at nesting depth 1). The file also uses `conde`/`fresh` without defining them, so it needs a microKanren prelude in addition to the fix. |
| `lib/golog/constrained_golog.pl` | Loads with syntax errors at lines 80 (`fd_in(V, Lo..Hi)`: `..` is not an operator without clpfd) and 247; those clauses are skipped. `demo_blocks` and `demo_coffee` run (energy 97, time 2; coffee delivered at depth 5). |
| `lib/golog/edaulc_agent.pl` | The README command (`swipl -g ... /dev/null`) fails because it passes `/dev/null` as a script. `swipl -g run_tests -t halt lib/golog/edaulc_agent.pl` runs it: **17/17 plunit tests pass**. |
| `quantum-holographic/sgmt-crystallization.pl` | Phases 1-2 print. Phase 3 raises `Unknown procedure: state_vector_kernel/2` (called by `verify_constraint/2`, defined nowhere in this copy), so no EGGs are generated. |
| `quantum-holographic/submissions/*.xml` + `xslt-normalize.xsl` | agent-1 is well-formed and normalizes (4,918 bytes). agent-2 (lines 60, 63, 92) and agent-3 (lines 65, 114) are not well-formed: they contain a raw `<` in embedded rule text outside CDATA, so xsltproc rejects them. The XSL itself is well-formed. |
| `dyna_golog_viterbi.ecl`, `.oz` | Not run (no runtime available). |
| `ruby-family-stack/spec/FORMAL_INVARIANTS.md` I20, I21 on CRuby 3.3.8 | I20 holds (`1/2 = 0`, `-1/2 = -1`, `1.0/2 = 0.5`). In I21, "Float + Rational -> Rational" does not hold on CRuby 3.3.8: `1.5 + Rational(1,2)` is a `Float`. |

## 4. Findings on the phi arithmetic files

These apply only to `lib/ast/phi_algebra.rb`, `lib/ast/rewrite.rb`,
`lib/kernel/symbolic_kernel.rb`, `lib/stages/apl_verify.rb`,
`lib/stages/axiom_proof.rb` and the tests in `spec/kernel_spec.rb`. The
README's "Key Properties" section describes these files, so its claims
are checked against them alone.

| README claim | finding in these files | evidence |
|---|---|---|
| "Canonical TRS: 388.985128" | Correct to the printed 6 decimals | Exactly, TRS = (534 + 872 phi)/5 = 388.98512763798166... `phi_algebra.rb` (f64) gives 388.98512763798175. `kernel_spec.rb` accepts a tolerance of 0.01 and `verify!` a tolerance of 0.001. |
| "Galois norm: TRS x sigma(TRS) in Q" | True mathematically; the code computes it in f64 | Exact norm = -1916/5 = -383.2. `symbolic_kernel.rb#norm` gives -383.2000000000001. The "norm is rational" test in `kernel_spec.rb` checks that the value is a finite Float. |
| "Confluence ... proven by exhaustive check" | `rewrite.rb#confluent?` compares two rule orders (forward, reversed) on the given term | It is called only on the canonical tree (`symbolic_kernel.rb`, `axiom_proof.rb`). No critical-pair or all-orderings check was found in the files read. The gated copy may contain more. |
| "phi^2 = phi + 1" normalization | `rewrite.rb` rewrites exponent-2 nodes only | On the canonical tree, `phi^3`..`phi^7` remain after `normalize`. `normalize(phi(4))` returns `(phi 4)`. |
| Stage 2 "APL geometric verification" | `apl_verify.rb` re-sums per-symbol subtrees with the same Ruby f64 evaluator | No APL interpreter is invoked in this copy. |
| Stage 3 "AXIOM formal proof" | Without a proof file, `axiom_proof.rb` reports `verified` = `Rewrite.confluent?` | The proof-file branch counts `sorry` occurrences. No proof file is in this copy. |

`lib/swarm/self_synthesis.rb` also emits a Self `verify` method as text.
That method compares `trs value + conjugate value` (the trace, which is
exactly 388) with 388.985128 within 0.001. The emitted Self code was not
executed; this is a reading of the generated text only.

## 4a. Candidates for the deterministic-host treatment

These candidates would follow the pattern used for DSS and phi. The
model emits only an intent. The host binds arguments from a strict
prompt template, runs an exact implementation, and returns the result
verbatim. Each candidate also needs a cross-check against the repo file
it comes from. The list is ordered by how well each fits.

1. **Viterbi max-product decoding** (`dyna_golog_viterbi.pl`, `_v1.py`;
   `.ecl`/`.oz` once they can be run).
   - The answer is a small exact object: the argmax path and its weight
     under the max-product (Viterbi) semiring. It can be computed
     exactly over rationals.
   - The Prolog and Python files already agree on toy weather
     (0.015552), so a three-way cross-check is available.
   - Caveat: the `_v1.py` lattice test expects the wrong path (see 3).
2. **EDAULC permission policy** (`edaulc_agent.pl`). Example intent:
   `execution_decision(Task, Role)` -> `permit` / `deny(Reason)`.
   - It is finite, closed-world and default-deny, and its 17 tests
     pass.
   - The answer set is enumerable, so an exhaustive Rust<->Prolog
     comparison is possible.
3. **Governance state machine and WORM chain**
   (`lib/kernel/governance.rb`, `lib/kernel/worm.rb`).
   - Possible intents: "is transition A -> B allowed", and "is this
     chain valid / verify entry i".
   - Both are a finite automaton plus SHA-256, so the answers are
     exact.
4. **GOLOG situation-calculus projection** (`holographic_golog_engine.py`;
   `constrained_golog.pl` demos; `.scm` after its paren fix).
   - Example intent: "fluent F after action sequence A from S0", or
     "is A executable" (Poss), over the Blocks, Elevator and Coffee
     domains.
   - Deterministic for the online/first-solution semantics. Its failing
     `test_star_and_test` should be resolved first.
5. **Ruby numeric and dispatch invariants**
   (`ruby-family-stack/spec/FORMAL_INVARIANTS.md` I20, I21, I9, I4).
   - Possible intents: "result/class of `a op b`" and "MRO of a given
     class/module layout".
   - The ground truth is a pinned CRuby, not a reimplementation. The I21
     line found above shows why the doc text cannot itself be the
     oracle.
   - The IR invariants IR1-IR5 (`ir/INTERMEDIATE_REPRESENTATION.md`)
     are checkable only once an IR exists. None is in this copy.
6. **Agent protocol state machines** (`agents_framework.rb`,
   `protocol_enforcement.rb`).
   - Possible intents: acyclic-authorization and consensus-threshold
     checks.
   - Only usable after the recursive-lock deadlock is fixed, because
     most of their tests cannot currently run.
7. **Not candidates in this copy:**
   - `sgmt-crystallization.pl`: the constraint kernel it calls is
     missing.
   - The `quantum-holographic` XML agents 2 and 3: not well-formed.
   - `self_synthesis.rb` and `topology.rb`: text and graph emission,
     with nothing to verify beyond structure.

## 5. Exact Rust port (`dss/host/src/phi.rs`)

The port works in Z[phi] with `i128` and checked arithmetic. Overflow
returns an error, and there is no f32/f64 anywhere in the module. It uses:
- phi^k = F(k-1) + F(k) phi;
- (a + b phi)(c + d phi) = (ac + bd) + (ad + bc + bd) phi;
- conj(a + b phi) = (a + b) - b phi;
- norm = a^2 + ab - b^2.

Unit tests check phi^2 = 1 + phi, x * conj(x) = norm(x), and
norm(534 + 872 phi) = -1916 * 5.

**Cross-check against Ruby** (`kernels/phi/phi_crosscheck.py`, output in
`kernels/phi/crosscheck_results.txt`). Ruby's own `PhiAlgebra.evaluate` in
`lib/ast/phi_algebra.rb` (f64) is compared with the Rust exact result evaluated to 60 digits.
Agreement means a relative error of at most 1e-9 of the summed term
magnitudes. There are 500 random inputs per row:

| inputs | normalize | mul | conj | norm |
|---|---|---|---|---|
| small (abs(coef) <= 50, k <= 12) | 500/500 | 500/500 | 500/500 | 500/500 |
| large (abs(coef) <= 10^9, k <= 60) | 500/500 | 500/500 | 500/500 | 500/500 |

The maximum relative error is 2.1e-15. However, Ruby's f64 cannot return
the exact integer norm for large inputs. Rounding Ruby's norm gives the
exact integer in **500/500 small cases but only 5/500 large cases**, which
is why the kernel answers come from the exact Rust port. This is a
property of f64 evaluation in `phi_algebra.rb`, not of any other file.

## 6. Intents and host routing

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

## 7. Data and model

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

## 8. Live llama.cpp results through the Rust host (host returns the kernel answer)

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
