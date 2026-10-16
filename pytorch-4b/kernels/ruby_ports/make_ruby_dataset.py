#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Training / eval data for the four ruby-kernel intents in dss-host.

Answers come from `dss-host answer` (Rust ports of dyna_golog_viterbi.pl,
edaulc_agent.pl, governance.rb, worm.rb). No Python kernel is used as an
oracle; the Python here is only the data harness.

Writes into pytorch-4b/dss/data/:
  ruby_train.jsonl              2000 per task
  ruby_test.jsonl               100 per task (hash-held-out, same distrib)
  ruby_test_ood.jsonl           100 per task (longer / larger / rarer)
  test_ruby_mixed_balanced.jsonl = prior mixed balanced + ruby_test
  test_ruby_mixed_ood.jsonl      = prior mixed OOD + ruby_test_ood
"""
import hashlib, json, pathlib, random, subprocess, string

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent.parent / "dss/data"
HOST = HERE.parent.parent / "dss/host/target/release/dss-host"
TASKS = ("viterbi", "permission", "governance", "worm_seal")
rng = random.Random(20261014)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(str(HERE.parent.parent / "tokenizer/tokenizer.json"))
MAX_PROMPT_TOK = 230  # prompt + " [INTENT: ...]" must fit the 256-token context


def fits(p):
    return len(TOK.encode(p + " [INTENT: VITERBI_DECODE]").ids) <= MAX_PROMPT_TOK

VOCAB = {
    "toy_weather": ["walk", "shop", "clean"],
    "pos_tagger": ["the", "a", "an", "cat", "dog", "man", "woman", "park", "house", "car",
                   "saw", "walked", "ate", "ran", "is", "in", "on", "with", "by", ".", "!", "?"],
    "cpg_island": ["A", "C", "G", "T"],
}
TASK_ATOMS = [
    "configure_aws_infrastructure", "design_ci_cd_pipeline", "draft_architecture_decision_record",
    "design_team_topology", "draft_stakeholder_memo", "diagnose_illness", "prescribe_treatment",
    "draft_contract", "interpret_statute", "recommend_stock_allocation", "prepare_tax_strategy",
    "summarize_public_topic",
]
ROLE_ATOMS = ["devops_specialist", "system_architect", "communications_lead", "org_designer"]
EXTRA_ATOMS = ["colonize_mars", "wizard", "sa_representative", "devops", "safety", "edaulc", "a", "x9"]
STATES = ["init", "verified", "sealed", "rejected"]
GENESIS = "0" * 64


def split(p):
    return "test" if int(hashlib.sha256(p.encode()).hexdigest(), 16) % 10 == 0 else "train"


def answer(prompts):
    # Batch through dss-host; answers are authoritative (Rust = Prolog/Ruby ports).
    out = subprocess.run([str(HOST), "answer"], input="\n".join(prompts) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(prompts), (len(out), len(prompts))
    bad = [o for o in out if o.startswith("ERROR")]
    assert not bad, bad[:5]
    return out


def atom(rng, pool):
    return rng.choice(pool)


def viterbi_prompt(rng, ood=False):
    model = rng.choice(list(VOCAB))
    voc = VOCAB[model]
    if ood:
        n = rng.randint(20, 32) if model == "cpg_island" else rng.randint(12, 24)
    else:
        n = rng.randint(1, 8 if model != "cpg_island" else 16)
    obs = " ".join(rng.choice(voc) for _ in range(n))
    return f"viterbi {model} {obs} ->"


def permission_prompt(rng, ood=False):
    if ood:
        # Unknown atoms or role/task swapped-class combinations are still
        # well-formed permission prompts; the host returns deny(...).
        t = atom(rng, TASK_ATOMS + EXTRA_ATOMS + ROLE_ATOMS)
        r = atom(rng, ROLE_ATOMS + EXTRA_ATOMS + TASK_ATOMS)
    else:
        # Prefer known KB pairs (covers permit and every deny class).
        if rng.random() < 0.85:
            t, r = atom(rng, TASK_ATOMS), atom(rng, ROLE_ATOMS)
        else:
            t, r = atom(rng, TASK_ATOMS + EXTRA_ATOMS), atom(rng, ROLE_ATOMS + EXTRA_ATOMS)
    return f"permission {t} {r} ->"


def governance_prompt(rng, ood=False):
    n = rng.randint(4, 8) if ood else rng.randint(1, 5)
    steps = " ".join(rng.choice(STATES) for _ in range(n))
    return f"governance steps {steps} ->"


def payload(rng, ood=False):
    if ood:
        # Larger but still ctx-safe canonical JSON.
        keys = [f"k{i}" for i in range(rng.randint(3, 6))]
        parts = []
        for k in keys:
            kind = rng.choice(["int", "str", "bool", "null", "obj"])
            if kind == "int":
                parts.append(f'"{k}":{rng.randint(-10**9, 10**9)}')
            elif kind == "str":
                s = "".join(rng.choice(string.ascii_letters + " _:-/") for _ in range(rng.randint(4, 24)))
                s = s.replace("\\", "\\\\").replace('"', '\\"')
                parts.append(f'"{k}":"{s}"')
            elif kind == "bool":
                parts.append(f'"{k}":{"true" if rng.random() < 0.5 else "false"}')
            elif kind == "null":
                parts.append(f'"{k}":null')
            else:
                parts.append(f'"{k}":{{"from":"{rng.choice(STATES)}","to":"{rng.choice(STATES)}"}}')
        return "{" + ",".join(parts) + "}"
    # Small canonical payloads (governance-style and trivial).
    return rng.choice([
        "{}",
        '{"a":1}',
        '{"a":1,"b":2}',
        '{"from":"init","to":"verified","evidence":{}}',
        '{"from":"verified","to":"sealed","evidence":{}}',
        '{"from":"init","to":"rejected","evidence":{}}',
        '{"ok":true}',
        '{"ok":false,"n":0}',
        '{"label":"stage:apl","n":3}',
        '[]',
        '[1,2,3]',
        '{"x":[1,null,true]}',
    ])


def worm_prompt(rng, ood=False, prev=None):
    label = rng.choice([
        "kernel:init", "stage:clojure", "stage:apl", "stage:axiom", "SOVEREIGN-FINAL",
        "governance:verified", "governance:sealed", "governance:rejected", "swarm:seal_0", "x", "a:b",
    ])
    if ood:
        label = label + ":" + "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 8)))
        if len(label) > 64:
            label = label[:64]
    y = rng.randint(2000, 2030); mo = rng.randint(1, 12); d = rng.randint(1, 28)
    h = rng.randint(0, 23); mi = rng.randint(0, 59); s = rng.randint(0, 59)
    ts = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:02d}Z"
    if prev is None:
        if ood and rng.random() < 0.5:
            prev = "".join(rng.choice("0123456789abcdef") for _ in range(64))
        else:
            prev = GENESIS
    pl = payload(rng, ood=ood)
    return f"worm seal label={label} ts={ts} prev={prev} payload={pl} ->"


def gen_task(task, rng, ood=False):
    if task == "viterbi":
        return viterbi_prompt(rng, ood)
    if task == "permission":
        return permission_prompt(rng, ood)
    if task == "governance":
        return governance_prompt(rng, ood)
    if task == "worm_seal":
        return worm_prompt(rng, ood)
    raise ValueError(task)


train, test, ood, seen = [], [], [], set()
FINITE = ("permission", "governance")  # prompt space too small for 2000 distinct rows
for task in TASKS:
    ntr = nte = 0
    tries = 0
    pool_train = []
    while (ntr < 2000 or nte < 100) and tries < 200000:
        tries += 1
        p = gen_task(task, rng, ood=False)
        if not fits(p):
            continue
        if split(p) == "test":
            if p not in seen and nte < 100:
                seen.add(p); test.append({"task": task, "prompt": p}); nte += 1
            continue
        if task in FINITE:
            pool_train.append(p); ntr += 1          # repeats allowed; test prompts never enter train
        elif p not in seen and ntr < 2000:
            seen.add(p); train.append({"task": task, "prompt": p}); ntr += 1
    if task in FINITE:
        for p in pool_train[:2000]:
            train.append({"task": task, "prompt": p}); seen.add(p)
    print(task, "train", min(ntr, 2000), "distinct_train", len(set(pool_train)) if task in FINITE else min(ntr, 2000), "test", nte, flush=True)
    n = 0
    while n < 100:
        if task == "permission":
            # OOD: novel atoms never in the KB or training pool
            w = lambda: "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 12))) + rng.choice(["", "_ops", "_review", "_x"])
            p = f"permission {rng.choice([w(), rng.choice(TASK_ATOMS)])} {w()} ->"
        else:
            p = gen_task(task, rng, ood=True)
        if p in seen or not fits(p):
            continue
        seen.add(p)
        ood.append({"task": task, "prompt": p}); n += 1

print(f"prompts: train={len(train)} test={len(test)} ood={len(ood)}", flush=True)


def fill(rows, tag):
    # Answer in batches of 200 to keep host IO bounded.
    for i in range(0, len(rows), 200):
        chunk = rows[i:i + 200]
        ans = answer([r["prompt"] for r in chunk])
        for r, a in zip(chunk, ans):
            r["answer"] = " " + a
        print(f"  {tag} {min(i+200, len(rows))}/{len(rows)}", flush=True)


fill(train, "train"); fill(test, "test"); fill(ood, "ood")
rng.shuffle(train)


def dump(name, rows):
    path = DATA / name
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)})", flush=True)


dump("ruby_train.jsonl", train)
dump("ruby_test.jsonl", test)
dump("ruby_test_ood.jsonl", ood)

# Mix with prior DSS+phi mixed sets (preserve their order, append ruby).
bal = [json.loads(l) for l in open(DATA / "test_mixed_balanced.jsonl")] + test
oodm = [json.loads(l) for l in open(DATA / "test_mixed_ood.jsonl")] + ood
dump("test_ruby_mixed_balanced.jsonl", bal)
dump("test_ruby_mixed_ood.jsonl", oodm)

# Stats
from collections import Counter
for name, rows in [("train", train), ("test", test), ("ood", ood)]:
    print(name, dict(Counter(r["task"] for r in rows)))
print("mixed balanced", len(bal), "mixed ood", len(oodm))
