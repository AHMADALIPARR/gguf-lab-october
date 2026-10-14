# dss-host

This is the orchestrator between a llama.cpp model and a CPU DSS kernel
written in Rust. The box has no GPU, so the kernel is plain CPU Rust. The
license is AGPL-3.0-only.

```
cargo build --release --offline
target/release/dss-host run  --mode intent --gguf MODEL.gguf "is DSS 101 202 303 ->"
target/release/dss-host eval --mode intent --gguf MODEL.gguf prompts.txt out.tsv
target/release/dss-host kernel < queries.txt     # e.g. "isdss 1 2 4", "bound 25 313"
```

## Modes

| mode | model emits | final answer |
|---|---|---|
| `intent` (default) | `[INTENT: CHECK_IS_DSS]`, `CARD_FROM_SUM`, `NEXT_GREEDY`, or `CHECK_BOUND`; generation stops there | the Rust kernel result, verbatim |
| `intent-gen` | the same intent | written by the model after `<result>` (round 5) |
| `args` | `<call>TOOL ARGS</call>` with the model's own arguments | written by the model |
| `gbnf` | as `args`, constrained by a GBNF grammar (llama-completion) | written by the model |

In `intent` mode the host parses the prompt with a strict per-task parser
(`src/parse.rs`) before the model is run. It then checks that the intent
names the prompt's own task and runs the kernel (`src/kernel.rs`). If any
step fails, it prints an explicit `ERROR: ...` and returns no answer. It
never guesses. The failure cases are:
- the prompt is rejected;
- the intent is missing or malformed;
- the intent names a different task;
- the kernel rejects the arguments.

Examples (live, `dss-intent-aug.gguf`):

```
$ dss-host run --mode intent --gguf dss-intent-aug.gguf "bound k 25 a 313 ->"
bound k 25 a 313 -> [INTENT: CHECK_BOUND] <result>b=313; 313>=313 yes</result> b=313; 313>=313 yes
$ dss-host run --mode intent --gguf dss-intent-aug.gguf "is DSS 101 202 303 ->"
is DSS 101 202 303 -> [INTENT: CHECK_IS_DSS] <result>no 303=101+202</result> no 303=101+202
$ dss-host run --mode intent --gguf dss-intent-aug.gguf "is DSS 202 101 ->"
ERROR: prompt rejected: set must be strictly increasing (sorted, no duplicates) (model emitted "")
```

The kernel agrees with `oracle.py` and `runtime/greedy.mjs` on 3302/3302
test queries (`../host_crosscheck.py`). Live results are in
`../../FEEDBACK_LOOP.md` (rounds 5 and 6).

## Ruby-kernel intents (no model trained on them yet)

Four further intents are ported from sovereign-ruby-kernel (commit
`52ff2a0`; the public copy may be partial because of its clone gate):

| intent | source file | example prompt | example answer |
|---|---|---|---|
| `VITERBI_DECODE` | `lib/golog/dyna_golog_viterbi.pl` | `viterbi toy_weather walk shop walk ->` | `path=rainy,rainy,rainy weight=243/15625` |
| `CHECK_PERMISSION` | `lib/golog/edaulc_agent.pl` | `permission draft_contract devops_specialist ->` | `deny(prohibited_class(advisory_legal))` |
| `GOVERNANCE_STEP` | `lib/kernel/governance.rb` | `governance steps verified sealed init ->` | `StateError: sealed → init forbidden` |
| `WORM_SEAL` | `lib/kernel/worm.rb` | `worm seal label=governance:verified ts=2026-10-14T14:18:00Z prev=<64 zeros> payload={"from":"init","to":"verified","evidence":{}} ->` | `18b7c117...3e22ef37` |

Parity is checked against the original runtimes (swipl, `ruby -rtime`) by
`dss-host xcheck ...` and `kernels/ruby_ports/run_parity.sh`. The counts
and the documented divergences are in `kernels/RUBY_KERNEL_PORTS.md`.
