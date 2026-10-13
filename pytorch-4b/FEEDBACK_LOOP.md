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
