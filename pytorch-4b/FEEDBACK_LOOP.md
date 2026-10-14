# Oracle feedback loop on DSS tasks

This note describes a feedback loop that fine-tunes the 34M CPU model
(`runs/small-cpu`, see TRAINING_RUN.md) on tasks graded by a deterministic
verifier ported from `runtime/greedy.mjs` in
SNAPKITTYWEST/sovereign-agi-kernel. All numbers below are measured; raw
per-round logs are in `dss/results/`.

## The property

`greedy.mjs` calls a set of positive integers "DSS" when any two subsets with
the same sum have the same size, so a subset's sum determines its
cardinality. The file header calls this "Erdos-Straus"; that name refers to an
unrelated conjecture about 4/n and is not used here.

`computeGreedySequence(k)` starts at `[1]` and repeatedly appends the smallest
integer larger than the last element that keeps the set DSS.

## Verifier (`dss/oracle.py`)

`greedy.mjs` rebuilds all 2^n subsets for each candidate. `oracle.py` keeps a
table `card[s]` (size of a subset with sum `s`, or -1) and updates it like a
0/1 knapsack. A candidate `x` is accepted iff for every reachable sum `s`,
`card[s+x]` is unreachable or equals `card[s]+1`. Each check costs
O(sum of the set) in numpy.

`dss/crosscheck.py` runs `greedy.mjs` under node and compares:
- `computeGreedySequence(14)`: identical
- `verifyQuadraticBound` on those 14 terms: identical
- `isDSS` on 2000 random sets (1133 DSS): 0 disagreements
- `buildCardinalityTable` on the DSS sets among the first 50: all entries agree
- every non-DSS set gets a collision witness (equal sums, different sizes),
  which is checked

### Greedy terms (`dss/greedy_terms.json`)

Computed with the oracle on one thread:

| k | 1 | 2 | 3 | 4 | 5 | ... | 16 | 17 | 18 |
|---|---|---|---|---|---|-----|----|----|----|
| a_k | 1 | 2 | 4 | 8 | 16 | ... | 32768 | 65536 | 131072 |

So a_k = 2^(k-1) for k = 1..18. The cumulative time was 4.9 s at k=16,
22.5 s at k=17, and 98.8 s at k=18. The run was stopped during k=19 because
each term costs about 4x the one before. The pattern is expected: starting
from `[1]`, if the set is {1, 2, ..., 2^(k-2)}, every integer below 2^(k-1)
is a subset sum of size at least 2, or is already in the set. Adding any such
integer creates a sum collision between subsets of different sizes, so the
smallest valid candidate is 2^(k-1).

`verifyQuadraticBound` checks a_k >= floor((k^2+1)/2). That bound **fails at
k = 3** (4 < 5) and holds for every other computed k.

## Dataset (`dss/make_dataset.py`, `dss/data/`)

The tasks use one-line prompts. Each stored answer is produced by the oracle
and then checked again by `dss/grade.py` before it is written.

| task | prompt -> answer | train | test |
|------|------------------|------:|-----:|
| next  | `DSS greedy next after 3 5 6 -> 12` (greedy step from a random DSS prefix) | 6000 | 100 |
| isdss | `is DSS 3 5 6 -> no 6=1+5` / `yes` | 6000 | 100 |
| card  | `DSS W 1 2 4 9 sum 11 -> size 2` | 6000 | 100 |
| bound | `bound k 5 a 16 -> yes 16>=13` | 4322 | 100 |

Training on sovereign-agi-kernel material (source, README, and derived data) was authorized by the licensor, the trustee of the kernel's licenses.

`kernel_text.jsonl` also contains the kernel's `README.md` and
`runtime/greedy.mjs`. During SFT they are mixed in as replay text.

The split is made by hashing each example's underlying set (or (k, a) for
bound), so the test set shares no sets with the training set. There are 0
prompt overlaps. The test set is fixed and used greedily decoded in every
round. Majority-class baselines on the test set are 0.93 for isdss (always
"yes"; only 7/100 test sets are non-DSS) and 0.40 for card (always "1"), so
isdss accuracy carries little signal.

## Loop (`dss/loop.py`)

1. **SFT warm start**: 400 steps, batch 16, loss on answer tokens only, lr
   3e-4. Each step also takes 2 replay windows of 128 tokens from
   `data/tokens.bin` (30% kernel text), weighted 0.5.
2. **Expert iteration / rejection sampling**: each round samples 4 answers
   (T=1.0) for 200 training prompts. The verifier grades them, and only
   verified-correct samples join the fine-tune pool. Then 150 SFT steps at
   lr 1e-4. There are 2 rounds.
3. **RL with verifiable rewards**: GRPO-style group-relative advantages
   (group of 4 samples per prompt, 8 prompts per update, 10 updates per
   round, 3 rounds, lr 2e-5). The reward is 1 if verified correct, 0 if
   wrong, and -0.1 if the output is malformed. There is a per-token KL penalty
   (k3 estimator, beta 0.05) against a frozen copy of the model taken after
   expert iteration (the pre-RL reference).

The reward comes from a deterministic verifier (`grade.py` + `oracle.py`),
not from a learned preference or reward model. This is reinforcement learning
with verifiable rewards, which is the form of RL-from-automated-feedback
(RLAIF) used here. The grader checks answers against the prompt itself, not
against stored strings, so any valid collision witness is accepted.

After every round the loop logs held-out accuracy and the mean next-token
loss on the first 64 windows (256 tokens each) of the old `data/val.bin`, to
track forgetting. All runs use 2 CPU threads.

## Results: prototype from `runs/small-cpu/best.pt`

Seed 0, wall time 17 min (07:44-08:02 PT, Oct 13 2026).

| stage | round | next | isdss | card | bound | all | old-val loss | extra |
|-------|------:|-----:|------:|-----:|------:|----:|-------------:|-------|
| base | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 3.2750 | |
| SFT | 0 | 0.74 | 0.90 | 0.65 | 0.06 | 0.588 | 3.8054 | |
| expert iter | 1 | 0.81 | 0.86 | 0.69 | 0.13 | 0.623 | 3.6660 | 445/800 samples verified |
| expert iter | 2 | 0.81 | 0.88 | 0.74 | 0.19 | 0.655 | 3.5800 | 419/800 verified |
| GRPO | 1 | 0.83 | 0.88 | 0.73 | 0.19 | 0.658 | 3.5812 | reward 0.479, KL 0.0064 |
| GRPO | 2 | 0.82 | 0.89 | 0.73 | 0.17 | 0.653 | 3.5806 | reward 0.485, KL 0.0058 |
| GRPO | 3 | 0.84 | 0.89 | 0.71 | 0.18 | 0.655 | 3.5801 | reward 0.511, KL 0.0155 |

Observations:
- SFT and expert iteration account for nearly all of the gain. The three
  GRPO rounds stay within +/-0.5 points overall. With 400 test items, one
  standard error is about 2.4 points, so RL has no measurable effect at this
  budget (30 updates).
- Old-val loss rose by 0.53 after SFT (forgetting) and recovered to +0.31
  over base by the end. The replay windows during expert iteration seem to
  help, but this was not ablated.
- The bound task stays weak (0.19). It requires computing floor((k^2+1)/2)
  in the answer string.

## Results: from `runs/small-cpu-v2/final.pt`

Same settings and seed, but with 8 threads (after v2 training finished).
final.pt was used instead of best.pt because it has the lower full-split val
loss. Run time was 10:20-10:35 PT on Oct 13 2026. The base evaluation shared
the CPU with a separate eval job. Log: `dss/results/v2-final-metrics.jsonl`.

| stage | round | next | isdss | card | bound | all | old-val loss | extra |
|-------|------:|-----:|------:|-----:|------:|----:|-------------:|-------|
| base | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 2.3332 | |
| SFT | 0 | 0.75 | 0.81 | 0.63 | 0.10 | 0.573 | 2.8826 | |
| expert iter | 1 | 0.78 | 0.85 | 0.74 | 0.21 | 0.645 | 2.8092 | 418/800 samples verified |
| expert iter | 2 | 0.79 | 0.88 | 0.78 | 0.20 | 0.663 | 2.8189 | 446/800 verified |
| GRPO | 1 | 0.81 | 0.87 | 0.78 | 0.21 | 0.668 | 2.8190 | reward 0.517, KL 0.0038 |
| GRPO | 2 | 0.84 | 0.88 | 0.74 | 0.20 | 0.665 | 2.8190 | reward 0.529, KL 0.0065 |
| GRPO | 3 | 0.84 | 0.88 | 0.76 | 0.20 | 0.670 | 2.8191 | reward 0.520, KL 0.0179 |

The pattern matches the prototype. SFT and expert iteration bring held-out
accuracy from 0 to 0.663. The GRPO rounds change it by less than one standard
error (about 2.4 points). Old-val loss ends 0.49 above the v2 base (2.333 ->
2.819), which is a clear forgetting cost at this replay ratio. The mean
training reward during RL ranged from 0.517 to 0.529; KL to the reference
grew from 0.004 to 0.018.

## Round 2: stepwise bound format, balanced is-DSS test, longer GRPO

### Bound diagnosis

Greedy outputs of the v2 loop's `rl.pt` on bound test items mostly had a
correct bound `b` (there are only 30 distinct k values). The errors came from
miscopying `a` (for example `yes 476>=392` for a=434), from wrong comparisons
(`no 8<8`), and from the verdict being emitted before any comparison.

### Data changes (`dss/make_dataset_v2.py`, `dss/data/stats_v2.json`)

- Bound uses a new format that states the bound before the verdict:
  `bound k 16 a 67 -> b=128; 67<128 no`. There are 10,000 train rows with
  k in 1..40, balanced at 5025 yes / 4975 no. The grader accepts both the
  old and new formats, but every number and the verdict must be correct.
- 4,000 extra non-DSS is-DSS train rows bring is-DSS training data to
  5037 yes / 4963 no.
- `test_balanced.jsonl` (500 items) contains next and card from the old
  test, 200 is-DSS items (100 yes / 100 no), and 100 bound items
  (50 yes / 50 no). The same hash split keeps it disjoint from training.
  The round-1 `test.jsonl` is kept unchanged for comparability.

### Run (`runs/dss-r2`, log `dss/results/r2-metrics.jsonl`)

- Start from the v2 loop's `rl.pt` (old test 0.670).
- SFT: 600 steps on `train_v2.jsonl`, lr 2e-4, 4 replay windows per step at
  weight 1.0.
- Expert iteration: 1 round.
- GRPO: 10 rounds x 30 updates = 300 updates. Group of 8, 8 prompts per
  update, lr 2e-5, KL beta 0.05 against the post-expert-iteration model.
  4 old-corpus replay windows per update, weight 0.5.
- Curriculum: tasks are sampled in proportion to their error on the
  balanced test set (floor 0.1).
- Threads: 4. Another worker's training job held the other cores for most
  of the run. Wall time 13:21-14:42 PT.

Accuracy is overall ± one binomial standard error. Per-task columns are
next / isdss / card / bound.

| stage | balanced all | balanced per task | old-test all | old-test per task | old-val loss | reward | KL |
|---|---|---|---|---|---|---|---|
| start (v2 loop rl.pt) | 0.562 ± 0.022 | .84/.515/.76/.18 | 0.670 ± 0.024 | .84/.88/.76/.20 | 2.8191 | | |
| SFT | 0.554 ± 0.022 | .79/.48/.58/.44 | 0.655 ± 0.024 | .79/.86/.58/.39 | 2.8415 | | |
| expert iter 1 | 0.628 ± 0.022 | .81/.51/.80/.51 | 0.720 ± 0.022 | .81/.82/.80/.45 | 2.8005 | | |
| GRPO 1 | 0.620 ± 0.022 | .82/.52/.75/.49 | 0.715 ± 0.023 | .82/.85/.75/.44 | 2.8012 | 0.534 | 0.0048 |
| GRPO 2 | 0.630 ± 0.022 | .82/.53/.77/.50 | 0.733 ± 0.022 | .82/.86/.77/.48 | 2.8027 | 0.535 | 0.0104 |
| GRPO 3 | **0.632 ± 0.022** | .81/.525/.77/.53 | 0.725 ± 0.022 | .81/.87/.77/.45 | 2.8039 | 0.500 | 0.0168 |
| GRPO 4 | 0.624 ± 0.022 | .82/.53/.74/.50 | 0.723 ± 0.022 | .82/.89/.74/.44 | 2.8050 | 0.551 | 0.0265 |
| GRPO 5 | 0.620 ± 0.022 | .82/.535/.73/.48 | 0.733 ± 0.022 | .82/.89/.73/.49 | 2.8054 | 0.586 | 0.0249 |
| GRPO 6 | 0.620 ± 0.022 | .82/.525/.74/.49 | 0.728 ± 0.022 | .82/.87/.74/.48 | 2.8066 | 0.584 | 0.0295 |
| GRPO 7 | 0.624 ± 0.022 | .82/.535/.75/.48 | 0.733 ± 0.022 | .82/.89/.75/.47 | 2.8079 | 0.501 | 0.0409 |
| GRPO 8 | 0.594 ± 0.022 | .82/.53/.63/.46 | 0.703 ± 0.023 | .82/.89/.63/.47 | 2.8090 | 0.535 | 0.0335 |
| GRPO 9 | 0.604 ± 0.022 | .82/.53/.68/.46 | 0.715 ± 0.023 | .82/.89/.68/.47 | 2.8102 | 0.530 | 0.0351 |
| GRPO 10 | 0.616 ± 0.022 | .82/.52/.76/.46 | 0.735 ± 0.022 | .82/.89/.76/.47 | 2.8120 | 0.564 | 0.0300 |

The best checkpoint by balanced accuracy is GRPO round 3 (`rl-r3.pt`):
0.632 balanced and 0.725 on the old test.

### Verdict

- **is-DSS has not been learned.** On the balanced test set the is-DSS score
  stays between 0.48 and 0.535 at every stage, which is chance for a 50/50
  yes/no split. The 0.82-0.89 on the old test reflects its 93% "yes" class
  balance.
- **Bound improved because of the format and data change, not RL.** Balanced
  bound accuracy went from 0.18 to 0.51 after SFT and expert iteration on
  the stepwise format. GRPO kept it between 0.46 and 0.53.
- **RL against the 0.663 baseline.** The selected checkpoint scores 0.725 on
  the old test. Against 0.663 the difference is 0.062 with a standard error
  of about 0.033 (z ≈ 1.9), which is not significant at the usual 2-sigma
  level. Most of that gap was already there before RL: expert iteration in
  this run reached 0.720. Compared with that pre-RL checkpoint, GRPO's best
  change is +0.004 on the balanced test (SE of the difference ≈ 0.031).
  **GRPO did not beat the expert-iteration baseline beyond noise.**
- **Old-val drift was small during this round.** With heavier replay, loss
  went from 2.819 to 2.804 at round 3 and 2.812 at round 10. It remains
  about 0.47 above the v2 base model (2.333), a gap inherited from round 1.
- KL to the reference grew to about 0.03-0.04 by rounds 7-10, while held-out
  accuracy did not improve.

### GGUF export (round 2, superseded)

The first export of `rl-r3.pt` declared the `default` pre-tokenizer. On DSS
prompts it matched PyTorch greedy answers in only 7 of 40 cases. That is
fixed in the next section.

## GGUF tokenizer fix

`tokenizer.json` has no normalizer and a ByteLevel pre-tokenizer with
`use_regex=true`, which is the GPT-2 split regex: each number is kept as one
` ?\p{N}+` chunk. llama.cpp's `gpt-2` pre type uses the same regex.
`export_gguf.py` now writes `tokenizer.ggml.pre = "gpt-2"` and refuses
tokenizer configs it does not recognise.

Token ids were compared with `dss/tokenizer_parity.py` (llama.cpp 4413a19
`llama-tokenize`, no BOS, against HF `tokenizers`):

| GGUF pre | DSS prompt+answer texts (500) | mixed: numbers, code, Markdown, unicode, whitespace (32) |
|---|---|---|
| `default` (old) | 0/500 identical | 21/32 |
| `gpt-2` (new) | **500/500** | **31/32** |

The one remaining mismatch is a `\r` byte. HF emits `<unk>` (id 0) for it
because the vocab has no token for that byte, and llama.cpp drops it.

Greedy answer parity (`dss/gguf_parity.py`, first answer line, 10 prompts
per task from the balanced test, PyTorch vs `llama-simple`):

| checkpoint | old GGUF (`default`) | new GGUF (`gpt-2`) |
|---|---|---|
| round 2 `rl-r3.pt` | 7/40 identical | 40/40 |
| round 3 `rl-r5.pt` | n/a | 40/40 |

## Round 3: is-DSS with written-out reasoning

### Data (`dss/make_dataset_v3.py`)

`train_v3.jsonl` is `train_v2.jsonl` (32,000 rows) with every is-DSS answer
(5037 yes / 4963 no) rewritten:

- non-DSS: ` 16=3+13 no`, i.e. two subsets with equal sum and different
  sizes from the oracle, then the verdict;
- DSS: ` sums 1:[6 12 15 20] 2:[18 21 26 27 32 35] 3:[33 38 41 47] 4:[53] yes`,
  i.e. all subset sums grouped by size. No sum appears in two groups, which
  is the DSS property.

Sets have 2..5 elements, and the longest prompt+answer is 58 tokens. The
grader scores only the verdict and, for "no", whether the witness is valid.
The certificate is not graded.

### Run (`runs/dss-r3`, log `dss/results/r3-metrics.jsonl`)

- Start from round 2's `rl-r3.pt`.
- SFT: 600 steps (lr 2e-4, 4 replay windows per step, weight 1.0).
- Expert iteration: 1 round.
- GRPO: 8 rounds x 30 updates. Group of 8, 8 prompts per update, KL beta
  0.05, 4 replay windows per update, curriculum on.
- max_new 64. Threads: 8; no other training was running at launch. Wall
  time 15:10-17:25 PT (cap 2.5 h).

Balanced is-DSS has n=200, so SE is about 0.035; balanced overall has n=500.

| stage | balanced is-DSS | balanced all | next / card / bound | old-test all | old-val loss | reward | KL |
|---|---|---|---|---|---|---|---|
| start | 0.525 ± 0.035 | 0.632 ± 0.022 | .81/.77/.53 | 0.725 ± 0.022 | 2.8039 | | |
| SFT | 0.500 ± 0.035 | 0.622 ± 0.022 | .81/.66/.64 | 0.713 ± 0.023 | 2.9871 | | |
| expert iter 1 | 0.510 ± 0.035 | 0.640 ± 0.022 | .83/.73/.62 | 0.758 ± 0.021 | 2.9666 | | |
| GRPO 1 | 0.525 ± 0.035 | 0.652 ± 0.021 | .83/.76/.62 | 0.763 ± 0.021 | 2.9640 | 0.668 | 0.0105 |
| GRPO 2 | 0.520 ± 0.035 | 0.644 ± 0.021 | .83/.72/.63 | 0.745 ± 0.022 | 2.9626 | 0.621 | 0.0082 |
| GRPO 3 | 0.540 ± 0.035 | 0.662 ± 0.021 | .83/.76/.64 | 0.760 ± 0.021 | 2.9626 | 0.572 | 0.0101 |
| GRPO 4 | 0.545 ± 0.035 | 0.654 ± 0.021 | .83/.71/.64 | 0.748 ± 0.022 | 2.9597 | 0.644 | 0.0167 |
| GRPO 5 | 0.545 ± 0.035 | **0.664 ± 0.021** | .84/.76/.63 | 0.763 ± 0.021 | 2.9566 | 0.661 | 0.0220 |
| GRPO 6 | 0.545 ± 0.035 | 0.664 ± 0.021 | .84/.75/.64 | 0.763 ± 0.021 | 2.9562 | 0.637 | 0.0164 |
| GRPO 7 | 0.550 ± 0.035 | 0.652 ± 0.021 | .83/.71/.62 | 0.748 ± 0.022 | 2.9557 | 0.607 | 0.0256 |
| GRPO 8 | 0.550 ± 0.035 | 0.656 ± 0.021 | .83/.73/.62 | 0.750 ± 0.022 | 2.9554 | 0.652 | 0.0385 |

The best checkpoint by balanced accuracy is GRPO round 5 (`rl-r5.pt`, tied
with round 6 and the earlier of the two). It was exported to
`runs/dss-r3/dss-r3-best.gguf` (not committed).

### Verdict

- **Graded is-DSS is still not learned.** The score is 0.50-0.55 at every
  stage. At its best (0.55) it is about 1.4 SE above chance, which is not
  significant.
- **The verdict itself is mostly right, but the witnesses are not.** On the
  200 balanced is-DSS items, `rl-r5.pt` gives the correct final verdict
  169 times (0.845): 91/100 DSS sets and 78/100 non-DSS sets. Only 18 of
  the 78 correct "no" answers carry a valid witness, so the graded score is
  109/200. The model's arithmetic in witnesses and certificates is wrong
  (for example `19=6+11` for {6, 11, 18, 19, 22}). It has learned what a
  collision looks like, not how to compute one.
- **Overall balanced accuracy, best checkpoint vs this run's
  expert-iteration checkpoint:** 0.664 vs 0.640. The difference is +0.024
  with an SE of about 0.030, so it is within noise.
- **Old-val loss rose by 0.15** (2.804 -> 2.956) during this round's SFT
  on the longer targets, despite heavier replay.

## Round 4: the model calls the kernel check as a tool (`dss/agent.py`)

### Format and harness

Each answer is one line that fits the 256-token context:

```
is DSS 3 5 6 -> <call>isdss 3 5 6</call> <result>yes</result> yes
bound k 12 a 60 -> <call>bound 12 60</call> <result>b=72; 60<72 no</result> b=72; 60<72 no
```

There are four tools: `isdss S`, `card W sum s`, `next S`, and `bound k a`.
`dss/agent.py` generates text up to `</call>`, runs the tool with
`oracle.py`, appends ` <result>...</result>`, and lets the model continue
until a newline. Only the text after the result is graded, by `grade.py`.

A call counts as malformed, with reward -0.1, if it doesn't parse, its
arguments are invalid, or it is missing.

Result tokens never contribute to the loss: they are masked to 0 in SFT,
in expert iteration, and in the GRPO policy-gradient and KL terms, so the
model is never trained to produce them itself. Segment boundaries are chosen
so that tokenizing the segments one by one gives the same ids as tokenizing
the whole line.

### Kernel equivalence

`agent.py nodecheck` compares the tool results with `runtime/greedy.mjs`
(`isDSS`, `cardinalityFromSum`) run under node, on every isdss and card
call in both test sets:
- isdss: 300/300 agree
- card: 200/200 agree

### Run (`runs/dss-agent`, log `dss/results/agent-metrics.jsonl`)

- Start from round 2's `rl-r3.pt`. On old-val loss it beats round 3's best
  (2.804 vs 2.957).
- Training prompts come from `train_v2.jsonl`.
- SFT: 500 steps, lr 2e-4, 4 replay windows per step at weight 1.0.
- Expert iteration: 1 round of 200 prompts x 8 samples.
- GRPO: 8 rounds x 30 updates. Group of 8, 8 prompts per update, KL beta
  0.05 against the post-expert-iteration model, 4 replay windows per
  update, curriculum on.
- 8 threads. Wall time 18:19-20:27 PT (cap 2.5 h).

Results on the balanced test (500 items). Each cell is well-formed call
rate / correct-tool rate / final accuracy ± SE. "Correct tool" means the
right tool with exactly the prompt's arguments.

| stage | next (100) | card (100) | isdss (200) | bound (100) | all (500) | old-val loss | reward | KL |
|---|---|---|---|---|---|---|---|---|
| start | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 2.8039 | | |
| SFT | .98/.98/.98±.014 | .98/.93/.95±.022 | 1.0/1.0/.985±.009 | .96/.71/.73±.044 | .984/.924/.926±.012 | 3.0447 | | |
| expert iter 1 | 1.0/1.0/1.0 | 1.0/.95/.96±.020 | 1.0/1.0/.995±.005 | .99/.73/.73±.044 | **.998/.936/.936±.011** | 3.0006 | | |
| GRPO 1 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/1.0/.99±.007 | .99/.70/.69±.046 | .998/.932/.928±.012 | 2.9996 | 0.766 | 0.0008 |
| GRPO 2 | 1.0/1.0/1.0 | 1.0/.94/.96±.020 | 1.0/1.0/.99±.007 | .99/.71/.70±.046 | .998/.930/.928±.012 | 2.9987 | 0.735 | 0.0058 |
| GRPO 3 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/1.0/.99±.007 | .99/.71/.71±.045 | .998/.934/.932±.011 | 2.9997 | 0.753 | 0.0108 |
| GRPO 4 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/1.0/.99±.007 | .99/.70/.70±.046 | .998/.932/.930±.011 | 2.9977 | 0.773 | 0.0131 |
| GRPO 5 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/.995/.99±.007 | .99/.70/.69±.046 | .998/.930/.928±.012 | 2.9975 | 0.766 | 0.0168 |
| GRPO 6 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/.995/.99±.007 | .99/.70/.71±.045 | .998/.930/.932±.011 | 2.9986 | 0.763 | 0.0124 |
| GRPO 7 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/.995/.99±.007 | .99/.70/.70±.046 | .998/.930/.930±.011 | 3.0004 | 0.762 | 0.0127 |
| GRPO 8 | 1.0/1.0/1.0 | 1.0/.96/.97±.017 | 1.0/.995/.99±.007 | .99/.70/.71±.045 | .998/.930/.932±.011 | 2.9977 | 0.786 | 0.0097 |

The best checkpoint by balanced accuracy is the expert-iteration one
(`ei.pt`).

### Comparison with round 3 (no tool)

| metric | round 3 best | tool use (ei.pt) |
|---|---|---|
| is-DSS | 0.545 ± 0.035 | 0.995 ± 0.005 |
| overall | 0.664 ± 0.021 | 0.936 ± 0.011 |
| old-val loss | 2.957 | 3.001 |

- With the tool, the remaining errors are almost all argument copying.
  For bound, the correct-tool rate (0.73) matches final accuracy (0.73);
  the model mostly miscopies a. For card, the correct-tool rate is
  0.95-0.96.
- GRPO did not improve on expert iteration: overall accuracy changed by
  -0.008 to -0.004, within one SE.
- Old-val loss rose by 0.20 over the starting checkpoint (2.804 -> 3.001)
  during tool-format SFT.

### llama.cpp tool loop

`agent.py llama CKPT GGUF` runs the same loop through `llama-simple`
(greedy): generate, cut at `</call>`, run the oracle tool, append the
result, and generate the final answer. It then compares the full
trajectory with the PyTorch greedy rollout.

The test used `ei.pt` exported to `runs/dss-agent/dss-agent-best.gguf`
(`gpt-2` pre-tokenizer; token parity 500/500 DSS texts and 31/32 mixed
texts). On 20 prompts (5 per task) the tool was called in 20/20 cases and
**20/20 trajectories were identical**: call, result, and final answer.

## Round 5: removing argument copy errors (intent mode, magnitude augmentation, GBNF)

A live llama.cpp test of `dss-agent-best.gguf` got 11 of 13 prompts right.
Both failures were copy errors: `bound k 25 a 313` produced the call
`bound 25 170`, and `is DSS 101 202 303` produced `isdss 19 26`.

### Rust host (`dss/host/`, crate `dss-host`, AGPL-3.0-only)

- **Kernel** (`src/kernel.rs`): a port of `oracle.py` that keeps an
  incremental sum -> subset-size table and finds collision witnesses in the
  same search order. There is no GPU on this box, so the kernel is plain
  CPU Rust.
- **Strict parser** (`src/parse.rs`): one template per task. It rejects
  irregular whitespace, non-ASCII text, signs, leading zeros, unsorted or
  duplicate sets, more than 8 elements, values above 10^6, and prompts that
  match no template or more than one. `cargo test` covers the rejection
  cases.
- **Driver** (`src/main.rs`): runs `llama-simple` (or `llama-completion`
  with a grammar in GBNF mode), runs the kernel, appends
  ` <result>...</result>`, and runs `llama-simple` again for the final
  answer.
- **Modes:**
  - `intent`: the model emits only ` [INTENT: CHECK_IS_DSS]`,
    `CARD_FROM_SUM`, `NEXT_GREEDY`, or `CHECK_BOUND`. The host binds the
    integers from the prompt and runs the kernel only if the intent names
    the prompt's own task.
  - `args`: the model's own arguments are used.
  - `gbnf`: the call is constrained to a tool name followed by integers.
- **Kernel cross-check** (`dss/host_crosscheck.py`): every tool query from
  the balanced, round-1 and OOD test sets, plus 1000 rows each from
  `train_v2` and `train_aug`, run through Rust, `oracle.py`, and
  `greedy.mjs` (verdict only for is-DSS, because `greedy.mjs` returns no
  witness). Agreement:

| query | Rust = oracle.py | Rust = greedy.mjs |
|---|---|---|
| isdss | 950/950 | 950/950 |
| card | 702/702 | 702/702 |
| next | 753/753 | 753/753 |
| bound | 897/897 | 897/897 |

Python is used only for training and for offline scoring of the host's
output TSVs (`dss/host_eval.py`, `dss/host_summary.py`). It is not used at
inference time.

### Data (`dss/make_dataset_ood.py`)

- `train_aug.jsonl`: 8000 prompts, 2000 per task. Numbers have 1-5 digits,
  about 75% of them 3-5 digits.
- `test_ood.jsonl`: 402 items, 100 per task with 3-5 digit numbers. is-DSS
  is 50 yes / 50 no (non-DSS sets get a planted a+b collision); bound is
  50/50 with k in 45..300. It also contains the two live-failure prompts.
  Keys are hash-split from training; there are no shared prompts.

### Models (both start from `ei.pt`; SFT 500 steps, then 1 expert-iteration round; 4 threads each, in parallel)

| model | training prompts | old-val loss |
|---|---|---|
| base `ei.pt` | (round 4) | 3.0006 |
| magnitude-aug (args) | train_v2 + train_aug | 2.9884 |
| intent | train_v2 + train_aug, intent format | 2.9818 |

Result tokens are masked from the loss in both. Logs:
`dss/results/{args,intent}-aug-metrics.jsonl`.

### Live llama.cpp results through the Rust host

All runs are greedy. Per-file numbers are in `dss/results/live/summary.jsonl`.
Column definitions:
- **copy err**: a call to the right tool with arguments that differ from
  the prompt.
- **final err**: the call was executed, but the final line is not the tool
  result.

**Balanced test (500):**

| config | well-formed | copy err | final err | next | card | is-DSS | bound | all ± SE |
|---|---|---|---|---|---|---|---|---|
| base `ei.pt`, args | 0.998 | 0.062 | 0.018 | 1.00 | 0.96 | 0.995 | 0.73 | 0.936 ± 0.011 |
| magnitude-aug, args | 1.000 | 0.024 | 0.006 | 1.00 | 1.00 | 1.000 | 0.87 | 0.974 ± 0.007 |
| intent | 1.000 | **0.000** | 0.042 | 1.00 | 1.00 | 1.000 | 0.79 | 0.958 ± 0.009 |

**OOD test (402, 3-5 digit numbers):**

| config | well-formed | copy err | final err | next | card | is-DSS | bound | all ± SE |
|---|---|---|---|---|---|---|---|---|
| base `ei.pt`, args | 0.761 | 0.761 | 0.376 | 0.00 | 0.31 | 0.396 | 0.00 | 0.177 ± 0.019 |
| magnitude-aug, args | 0.470 | 0.281 | 0.289 | 0.00 | 0.21 | 0.218 | 0.168 | 0.149 ± 0.018 |
| intent | 1.000 | **0.000** | 0.585 | 0.02 | 1.00 | 0.495 | 0.149 | **0.415 ± 0.025** |

On OOD, "well-formed" for the args configurations counts calls the host
could execute. A call with mis-copied arguments can still parse, which is
why well-formed and copy-error rates can both be high.

**GBNF vs plain args, base `ei.pt`.** llama-completion with a grammar took
about 20 s per call on this box, so GBNF was run on stratified subsets
(25 per task, `dss/data/gbnf_sub_*.jsonl`; OOD also includes the two
live-failure prompts), and plain args was scored on the same prompts:

| subset | config | well-formed | copy err | all ± SE |
|---|---|---|---|---|
| OOD (102) | args | 0.804 | 0.804 | 0.186 ± 0.039 |
| OOD (102) | args + GBNF | 0.951 | 0.951 | 0.216 ± 0.041 |
| balanced (100) | args | 0.990 | 0.100 | 0.900 ± 0.030 |
| balanced (100) | args + GBNF | 1.000 | 0.110 | 0.900 ± 0.030 |

With the grammar, every call parses, but the copy-error rate does not drop.
The grammar only constrains syntax, not which integers are written.
Accuracy is unchanged within SE.

**The two live-failure prompts:**
- base args: `bound 25 170`; `isdss 19 26`. Both wrong.
- magnitude-aug: `bound 25 313` is copied correctly, but the final answer
  says `361>=313`. `isdss 101 20269` is wrong.
- intent: the call is bound correctly and the kernel returns the right
  result (`b=313; 313>=313 yes`, `no 303=101+202`), but the model
  mis-transcribes it in the final answer (`619>=313`; `no 683=421+=True`).
  Both wrong.

### Parity (PyTorch greedy vs llama.cpp through the Rust host)

24 prompts: 3 per task from the balanced test and 3 per task from OOD.
- **Intent model:** 24/24 identical trajectories.
- **Magnitude-aug args model:** 24/24 identical generated calls. 21/24 full
  trajectories are identical. The other 3 are copy-error calls (an
  unsorted set, a repeated element, a value above 10^6). The strict Rust
  parser rejects them, while the more lenient Python harness executed them.
  The generated tokens are the same.

### Verdict

- **Intent mode removes argument copy errors: 0.000 on both test sets,
  versus 0.062 / 0.761 for base args.** OOD accuracy rises from 0.177 to
  0.415 ± 0.025.
- **The remaining failure is transcribing the result into the final
  answer.** For intent on OOD, 58.5% of rows have a final answer that
  differs from the kernel result, mostly in next (0.98) and bound (0.85).
  Card is 1.00 because its result is a single small number.
- **Magnitude augmentation helps on balanced (0.974) but not on OOD
  (0.149).** It reduces copy errors but often produces calls the strict
  parser rejects.
- **GBNF fixes syntax only.** It does not change copy errors or accuracy.
- The two live-failure prompts are still wrong in every configuration.
  The kernel result is right under intent mode; the model's final
  transcription is what fails.

A next step that has not been tried here: let the host emit the kernel
result as the answer, or train a final answer that only states the verdict
(the copy step is the remaining failure).
