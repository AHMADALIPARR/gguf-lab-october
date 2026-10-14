# Ruby-kernel ports in dss-host: Viterbi, EDAULC, governance + WORM

Three components of SNAPKITTYWEST/sovereign-ruby-kernel are ported to the
Rust host (`dss/host`). The source was the public clone at
`/workspace/sovereign-ruby-kernel`, commit `52ff2a0`. The model is **not**
trained on these intents yet; this round covers only the host and the
parity tests. All counts below come from
`kernels/ruby_ports/run_parity.sh`, whose output is saved in
`kernels/ruby_ports/parity_results.txt`.

**Clone gate.** The repository's `CLONE_GATE.md` says the repo is protected
by clone-gate verification, so the public copy may be partial. Everything
here applies only to the exact files named. Nothing is claimed about
components that may exist only in the gated copy.

## Scope: exact source files

| port (Rust) | source file(s) | what was ported | not ported |
|---|---|---|---|
| `src/viterbi.rs` | `lib/golog/dyna_golog_viterbi.pl` sections 1-3 and 7 | `viterbi_decode/4` (probability domain); the `toy_weather`, `pos_tagger` and `cpg_island` models | log-domain variant, lattice decoding, situation-calculus/GOLOG layer |
| `src/edaulc.rs` | `lib/golog/edaulc_agent.pl` | the knowledge base and `execution_decision/3`, clause order preserved | `validate_kb/0`, `permitted/1` instantiation-error behaviour |
| `src/governance.rs` | `lib/kernel/governance.rb`, `lib/kernel/worm.rb` | the `Gate` transition table and `StateError` message; `WORM.seal` hashing (SHA-256 over `JSON.generate`) | thread-safety (Mutex), timestamps from the clock, history list |

`lib/golog/dyna_golog_viterbi_v1.py` served as a second reference for the
Viterbi port. No Python is part of the deliverables. The `_v1.py`
comparison used an uncommitted scratch driver that imports the original
file unmodified; its numbers are reported separately below.

## Intents added (host returns the kernel result verbatim)

| intent | prompt template (strict) | answer |
|---|---|---|
| `VITERBI_DECODE` | `viterbi <toy_weather\|pos_tagger\|cpg_island> <obs ...> ->` (1..32 symbols from the model's vocabulary) | `path=rainy,rainy,rainy weight=243/15625` (exact rational) or `no path (all weights 0)` |
| `CHECK_PERMISSION` | `permission <task> <role> ->` (atoms `[a-z][a-z0-9_]{0,63}`) | the Prolog decision term, e.g. `permit`, `deny(prohibited_class(advisory_legal))` |
| `GOVERNANCE_STEP` | `governance steps <state ...> ->` (1..8 of init/verified/sealed/rejected, applied to a fresh gate at `init`) | final state, or `StateError: sealed → init forbidden` (Ruby's message) |
| `WORM_SEAL` | `worm seal label=<[A-Za-z0-9_:.-]{1,64}> ts=<YYYY-MM-DDTHH:MM:SSZ> prev=<64 hex> payload=<JSON> ->` | 64-hex SHA-256 seal |

Parser rules:
- **The payload must already be in canonical `JSON.generate` form.** It is
  re-generated and compared byte for byte. Floats and duplicate keys are
  rejected, as is anything else non-canonical.
- **The time is an input.** The host never reads the clock.
- **Rejections return `ERROR: prompt rejected: ...`.** This happens when a
  prompt fails its template or matches more than one.
- **The model must name the matching intent.** It emits only
  `[INTENT: ...]`, and the host compares it with the prompt's own task, the
  same rule as for DSS and phi.

## Parity against the original runtimes

Runtimes: SWI-Prolog 9.2.9 and Ruby 3.3.8 (`ruby -rtime`, since `worm.rb`
uses `Time#iso8601` without `require 'time'`). The drivers load the
original files unmodified, with two exceptions:
- **`viterbi_xcheck.pl` drops the source's final directive.** That
  directive is `:- initialization(demo, main).`, which would run the demo
  and halt.
- **`governance_xcheck.rb` stubs `Time.now`.** It returns the case
  timestamp.

**Original test suites, run unmodified:**
- `edaulc_agent.pl`: 17/17 plunit tests pass.
- `dyna_golog_viterbi.pl`: 4/4 tests pass ("ALL TESTS PASSED").
- `spec/kernel_spec.rb` with `-rtime`: 24/24 pass.

Rust unit tests restate the decision-level plunit tests, the
toy-weather/golog-wired Viterbi cases, and the WORM/Governance tests of
`kernel_spec.rb`: `cargo test` 6/6 pass.

### Viterbi (Rust exact) vs `dyna_golog_viterbi.pl` (swipl, floats)

| suite | cases | path agree | weight within 1e-12 rel | weight bit-equal | documented divergences | unexplained |
|---|---|---|---|---|---|---|
| source test/demo sequences | 8 | 8 | 8 | 2 | 0 | 0 |
| builtin models, random obs (len 1..26) | 3000 | 3000 | 3000 | 603 | 0 | 0 |
| random HMMs, probs 0.001..0.999 (+zeros) | 3000 | 2973 | 2987 | 1036 | 13 T=1 all-zero, 14 float tie-break | 0 |
| random HMMs, probs in {0, .25, .5, .75, 1} | 3000 | 2981 | 2981 | 2981 | 19 T=1 all-zero | 0 |

How paths are compared: positions t >= 1 are compared with Prolog. At
t = 0, Prolog's own output is an unbound variable (see the source bugs
below), and the comparison requires that.

The "float tie-break" cases are those where the exact computation has two
equal maxima (the first one in state order wins). In 14 such cases the
Prolog doubles differ in the last bit and pick the other candidate. In
every one of those 14 the weights still agree within 1e-12.

### Viterbi (Rust exact) vs `dyna_golog_viterbi_v1.py` (scratch driver, floats)

The same case files were used:

| suite | cases | path agree (full path incl. t = 0) | weight within 1e-12 | float tie-break divergences | unexplained |
|---|---|---|---|---|---|
| source sequences | 8 | 8 | 8 | 0 | 0 |
| builtin models | 3000 | 3000 | 3000 | 0 | 0 |
| random HMMs | 3000 | 2978 | 3000 | 22 | 0 |
| tie-heavy HMMs | 3000 | 3000 | 3000 | 0 | 0 |

`_v1.py` computes `prev * (trans * emit)`, while the Prolog file computes
`(prev * trans) * emit`. That is why the two float references break
different ties.

### EDAULC vs `edaulc_agent.pl`

The test set is exhaustive. It takes all 30 atoms (the 12 tasks, the 4
roles and 14 extra atoms such as `wizard`, `colonize_mars`, class names and
competency names) and pairs each atom with every other, as both task and
role.
- **Kernel:** 900 cases, **900/900 identical** decision terms.
- **Prompt level:** 900/900 identical.
- **No case had more than one solution.**

Decisions observed: 540 `deny(unknown_task)`, 312 `deny(unknown_role)`,
24 `deny(prohibited_class(_))` (8 per class), 15 `deny(out_of_domain)` and
9 `permit`.

### Governance and WORM vs `governance.rb` + `worm.rb`

- **Every step sequence of length 0..6** (5,461 sequences) was run at two
  timestamps.
  - Final state or `StateError` message, plus every WORM seal: **5461/5461
    identical** at each timestamp. That is 4,151 seals compared per
    timestamp.
  - The Ruby side also checked `WORM.valid?` and `verify_entry` for every
    entry.
- **Random WORM chains.** There were 2 × 3000 chains of 1-5 seals, with
  random labels, timestamps and nested JSON payloads. The payload strings
  include `"`, `\`, `/`, C0 controls, DEL, U+2028, CJK and emoji.
  - Seed 11: **8936/8936** seals identical.
  - Seed 12: **9036/9036** seals identical.
- **Prompt level:** 5460/5460 `governance steps` answers identical.

## Divergences from the source (deliberate, documented)

1. **Viterbi t = 0 state.** `viterbi_traceback/4` in
   `dyna_golog_viterbi.pl` returns a path whose first element is unbound
   (it prints as `[_12142,rainy,...]`). The time-0 state is never bound.
   The port returns the time-0 state from the time-1 backpointer, which
   matches `reconstruct_path` in `_v1.py`.
2. **Viterbi T = 1 with all weights 0.** Prolog returns `[_]` with weight
   `0.0`, while the port answers `no path (all weights 0)`. For T > 1 with
   all weights 0, Prolog fails and the port gives the same answer.
3. **Exact rationals instead of doubles.**
   - Inputs are read from their decimal text exactly, so `0.6` = 3/5.
     `pos_tagger`'s `1.0/5.0` is taken as 1/5.
   - On exact ties the first state in order wins. Float evaluation can
     break those ties differently, as counted above.
4. **WORM.** The timestamp is an argument rather than `Time.now`, and
   payloads are limited to strings, integers, booleans, null, arrays and
   objects. Floats are rejected rather than imitating Ruby `Float#to_s`.
   Governance seals use `evidence: {}`, the `Gate#transition!` default.
5. **EDAULC.**
   - `status(active)` is a constant in the source, so the
     `deny(agent_inactive)` branch is unreachable in both.
   - `deny(role_not_qualified)` and `deny(principle_violation(_))` are
     ported but unreachable with the shipped knowledge base.
   - Non-ground queries are not representable in the prompt template.

## Known source bugs (in the files named; not fixed upstream here)

- **`lib/golog/dyna_golog_viterbi_v1.py`, `test_lattice_decode`.** On its
  own lattice (0→1 0.6, 1→3 0.7, 0→2 0.4, 2→3 0.9), the test asserts the
  path `[0,2,3]` with weight 0.36. The max-product optimum is `[0,1,3]`
  with 0.42.
  - The test fails, and the remaining tests do not run: 5 tests pass
    before it.
  - The expectation is wrong, not the engine. The lattice decoder was not
    ported.
- **`lib/golog/dyna_golog_viterbi.pl`, `viterbi_traceback/4`.** The path
  has an unbound first element (divergence 1 above).
- **`lib/kernel/worm.rb`.** It calls `Time#iso8601` without
  `require 'time'`. Run as shipped, it raises `NoMethodError`.
- **`lib/golog/edaulc_agent.pl`.** The README's test command
  (`swipl -g "..." /dev/null`) fails because `/dev/null` is not a script.
  `swipl -g run_tests -t halt lib/golog/edaulc_agent.pl` works.

## Reproduce

```
cd pytorch-4b/dss/host && cargo build --release && cargo test --release
pytorch-4b/kernels/ruby_ports/run_parity.sh /workspace/sovereign-ruby-kernel /tmp/parity
```
