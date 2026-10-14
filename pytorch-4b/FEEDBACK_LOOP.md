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
