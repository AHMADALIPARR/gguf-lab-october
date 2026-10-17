#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Tool-calling harness: the model calls the DSS kernel check as a tool.

Trajectory format (one line, fits ctx 256):
  <prompt> <call>TOOL ARGS</call> <result>RESULT</result> FINAL
Tools (results from oracle.py, cross-checked against runtime/greedy.mjs):
  isdss 3 5 6          -> "yes" | "no A=B"   (A, B: equal sum, sizes differ)
  card 1 2 4 9 sum 11  -> subset size for that sum (-1 if unreachable)
  next 3 5 6           -> next greedy DSS term after the given prefix
  bound 12 60          -> "b=72; 60<72 no" | "b=B; a>=B yes"
The model generates up to "</call>"; the harness runs the tool, appends
" <result>...</result>" and the model continues to FINAL (until newline).
Only FINAL is graded (grade.py). Result tokens are never trained on: their
loss mask is 0 in SFT and in the policy-gradient term.

Subcommands:
  train      SFT -> expert iteration -> GRPO with the harness in the loop
  eval       evaluate a checkpoint (PyTorch) on a test file
  llama      run the same tool loop with llama-simple on a GGUF and compare
             with PyTorch greedy rollouts (parity)
  nodecheck  compare oracle tool results with greedy.mjs (node) on test sets
"""
import argparse, copy, json, os, pathlib, re, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from eval import load_model  # noqa: E402
from grade import grade, MALFORMED  # noqa: E402
from oracle import DSSTable, is_dss, collision_witness, quadratic_bound  # noqa: E402
import loop  # noqa: E402

EOS = 2
CALL_RE = re.compile(r"\s*<call>(isdss|card|next|bound) ([^<]*)</call>")
# Intent mode: the model names the check only; the host binds the integers
# from the prompt (expected_call is a strict per-task fullmatch parser).
INTENTS = {"isdss": "CHECK_IS_DSS", "card": "CARD_FROM_SUM", "next": "NEXT_GREEDY", "bound": "CHECK_BOUND",
           "phi_normalize": "PHI_NORMALIZE", "phi_mul": "PHI_MUL", "phi_conj": "PHI_CONJ", "phi_norm": "PHI_NORM",
           "viterbi": "VITERBI_DECODE", "permission": "CHECK_PERMISSION",
           "governance": "GOVERNANCE_STEP", "worm_seal": "WORM_SEAL"}
INTENT_TOOL = {v: k for k, v in INTENTS.items()}
INTENT_RE = re.compile(r"\s*\[INTENT: (" + "|".join(INTENTS.values()) + r")\]")


def register_kernel_answers(rows):
    """Answers for host-bound kernels come from the dataset rows (dss-host)."""
    import grade as G
    for r in rows:
        if r["task"].startswith("phi_") or r["task"] in ("viterbi", "permission", "governance", "worm_seal"):
            G.PHI_ANS[r["prompt"]] = r["answer"].strip()


def register_phi(rows):
    register_kernel_answers(rows)
MODE = ["args"]


# ---------------------------------------------------------------- tools
def run_tool(name, args):
    """Return the result string, or None if the arguments are malformed."""
    if name.startswith("phi_") or name in ("viterbi", "permission", "governance", "worm_seal"):
        # args is the prompt itself (host-bound); answer from the dataset/kernel
        import grade as G
        return G.PHI_ANS.get(args)
    try:
        xs = args.split()
        if name == "isdss":
            S = [int(x) for x in xs]
            if not S or len(S) > 8 or len(set(S)) != len(S) or min(S) <= 0: return None
            if is_dss(S): return "yes"
            a, b = collision_witness(S)
            if len(a) > len(b): a, b = b, a
            return f"no {'+'.join(map(str, a))}={'+'.join(map(str, b))}"
        if name == "card":
            i = xs.index("sum")
            W = [int(x) for x in xs[:i]]; s = int(xs[i + 1])
            if len(xs) != i + 2 or not W or len(W) > 8 or not is_dss(W): return None
            return str(DSSTable(W).cardinality(s))
        if name == "next":
            S = [int(x) for x in xs]
            if not S or len(S) > 8 or S != sorted(set(S)) or not is_dss(S) or S[-1] > 10**6: return None
            t = DSSTable(S); x = S[-1] + 1
            while t.conflict(x) is not None: x += 1
            return str(x)
        if name == "bound":
            k, a = int(xs[0]), int(xs[1])
            if len(xs) != 2 or k <= 0 or k > 1000: return None
            b = quadratic_bound(k)
            return f"b={b}; {a}>={b} yes" if a >= b else f"b={b}; {a}<{b} no"
    except (ValueError, IndexError):
        return None
    return None


def expected_call(prompt):
    if prompt.startswith("phi "):
        return "phi_" + prompt.split()[1], prompt
    if prompt.startswith("viterbi "):
        return "viterbi", prompt
    if prompt.startswith("permission "):
        return "permission", prompt
    if prompt.startswith("governance "):
        return "governance", prompt
    if prompt.startswith("worm seal "):
        return "worm_seal", prompt
    m = re.fullmatch(r"is DSS ([\d ]+) ->", prompt)
    if m: return "isdss", m.group(1)
    m = re.fullmatch(r"DSS W ([\d ]+) sum (\d+) -> size", prompt)
    if m: return "card", f"{m.group(1)} sum {m.group(2)}"
    m = re.fullmatch(r"DSS greedy next after ([\d ]+) ->", prompt)
    if m: return "next", m.group(1)
    m = re.fullmatch(r"bound k (\d+) a (\d+) ->", prompt)
    if m: return "bound", f"{m.group(1)} {m.group(2)}"
    raise ValueError(prompt)


def segments_for(prompt):
    """Gold trajectory as (text, trainable) segments."""
    name, args = expected_call(prompt)
    res = run_tool(name, args)
    call = f" [INTENT: {INTENTS[name]}]" if MODE[0] == "intent" else f" <call>{name} {args}</call>"
    return [(prompt, 0), (call, 1),
            (f" <result>{res}</result>", 0), (f" {res}\n", 1)]


# ---------------------------------------------------------------- rollout
@torch.no_grad()
def gen(model, ids, temp, max_new, stop, tok):
    """Generate token ids until stop(text) is true, newline, EOS or max_new."""
    out = []
    for _ in range(max_new):
        lg = model(torch.tensor([ids + out][-256:]))[0, -1]
        nxt = int(lg.argmax()) if temp <= 0 else int(torch.multinomial(torch.softmax(lg / temp, -1), 1))
        if nxt == EOS: break
        out.append(nxt)
        txt = tok.decode(out)
        if "\n" in txt or stop(txt): break
    return out


def rollout(model, tok, prompt, temp=0.0, max_call=40, max_final=24):
    """Returns dict: segments [(ids, trainable)], final text, call info."""
    p = tok.encode(prompt).ids
    intent = MODE[0] == "intent"
    c = gen(model, p, temp, max_call, (lambda t: "]" in t) if intent else (lambda t: "</call>" in t), tok)
    ctext = tok.decode(c)
    segs = [(p, 0), (c, 1)]
    info = {"call_text": ctext, "wellformed": False, "correct_tool": False, "copy_error": False, "result": None}
    exp = expected_call(prompt)
    if intent:
        m = INTENT_RE.fullmatch(ctext)
        name = INTENT_TOOL[m.group(1)] if m else None
        args = exp[1] if name == exp[0] else None  # bind only to the prompt's own task
    else:
        m = CALL_RE.fullmatch(ctext)
        name, args = (m.group(1), m.group(2)) if m else (None, None)
    if name and args is not None:
        res = run_tool(name, args)
        if res is not None:
            info["wellformed"] = True
            info["correct_tool"] = (name, " ".join(args.split())) == exp
            info["copy_error"] = name == exp[0] and " ".join(args.split()) != exp[1]
            info["result"] = res
            r = tok.encode(f" <result>{res}</result>").ids
            # The final answer copies the tool result; a fixed 24-token budget made
            # long results (worm_seal 64-hex ~40 tok, long viterbi paths) ungradeable.
            # DSS_FIXED_FINAL=1 restores the legacy fixed cap for comparison.
            if os.environ.get("DSS_FIXED_FINAL") != "1":
                max_final = max(max_final, len(tok.encode(f" {res}").ids) + 4)
            f = gen(model, p + c + r, temp, max_final, lambda t: False, tok)
            segs += [(r, 0), (f, 1)]
            info["final"] = tok.decode(f).split("\n")[0]
            info["reward"] = grade(prompt, info["final"])
            return segs, info
    info["final"] = None
    info["reward"] = MALFORMED  # no executable call
    return segs, info


def to_tensor(segs_list):
    seqs = [[t for s, _ in segs for t in s] for segs in segs_list]
    masks = [[m for s, m in segs for _ in s] for segs in segs_list]
    T = max(map(len, seqs))
    x = torch.full((len(seqs), T), EOS, dtype=torch.long); mk = torch.zeros(len(seqs), T)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s); mk[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return x, mk


def gold_segs(tok, prompt):
    return [(tok.encode(t).ids, m) for t, m in segments_for(prompt)]


# ---------------------------------------------------------------- eval
def evaluate_rows(model, tok, rows):
    by = {}
    for r in rows:
        _, info = rollout(model, tok, r["prompt"])
        by.setdefault(r["task"], []).append(info)
    out = {}
    allinfo = [i for v in by.values() for i in v]
    for name, infos in list(by.items()) + [("all", allinfo)]:
        n = len(infos)
        d = {"n": n}
        for k in ("wellformed", "correct_tool", "copy_error"):
            d[k] = round(sum(i[k] for i in infos) / n, 4)
        p = sum(i["reward"] == 1.0 for i in infos) / n
        d["acc"] = round(p, 4); d["acc_se"] = round((p * (1 - p) / n) ** 0.5, 4)
        out[name] = d
    return out


# ---------------------------------------------------------------- train
class Ctx(loop.Ctx):
    pass


def lm_step_loss(model, x, m):
    return loop.masked_ce(model, x, m)


def replay_loss(ctx, n, L):
    rb = loop.replay_batch(ctx, n, L)
    lg = model_ref[0](rb[:, :-1])
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), rb[:, 1:].reshape(-1))


model_ref = [None]


def sft(ctx, model, pool, steps, lr, tag):
    a = ctx.a
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train(); t0 = time.time(); tot = 0.0
    for step in range(1, steps + 1):
        x, m = to_tensor([p for p in (ctx.rng.sample(pool, a.batch))])
        loss = loop.masked_ce(model, x, m)
        (loss + a.replay_weight * replay_loss(ctx, a.replay, a.replay_len)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); tot += loss.item()
        if step % 25 == 0:
            print(f"[{tag}] step {step}/{steps} loss {tot/25:.4f} {time.time()-t0:.0f}s", flush=True); tot = 0.0
    model.eval()


def evaluate(ctx, model, stage, rnd, **extra):
    t0 = time.time()
    res = evaluate_rows(model, ctx.tok, ctx.test)
    if ctx.a.curriculum:
        ctx.task_w = {t: max(0.1, 1.0 - res.get(t, {"acc": 0})["acc"]) for t in ctx.by_task}
    ctx.log(stage=stage, round=rnd, heldout=res, old_val_loss=loop.old_val_loss(ctx, model, ctx.a.val_windows),
            eval_s=round(time.time() - t0), **extra)


def grpo_round(ctx, model, ref, steps, rnd):
    a = ctx.a
    opt = torch.optim.AdamW(model.parameters(), lr=a.rl_lr, weight_decay=0.0)
    rs, kls = [], []
    for step in range(steps):
        tasks = list(ctx.task_w)
        prompts = [ctx.rng.choice(ctx.by_task[t]) for t in
                   ctx.rng.choices(tasks, weights=[ctx.task_w[t] for t in tasks], k=a.rl_prompts)]
        for r in prompts:
            model.eval()
            outs = [rollout(model, ctx.tok, r["prompt"], a.temp) for _ in range(a.group)]
            rew = torch.tensor([i["reward"] for _, i in outs]); rs.extend(rew.tolist())
            adv = (rew - rew.mean()) / (rew.std() + 1e-4) if rew.std() > 0 else torch.zeros_like(rew)
            x, m = to_tensor([s for s, _ in outs])
            model.train()
            tok_lp, _, mm = loop.seq_logprobs(model, x, m)
            with torch.no_grad():
                ref_lp, _, _ = loop.seq_logprobs(ref, x, m)
            d = ref_lp - tok_lp; kl = torch.exp(d) - d - 1
            ntok = mm.sum(1).clamp(min=1)
            pg = -(adv[:, None] * tok_lp * mm).sum(1) / ntok
            klt = (kl * mm).sum(1) / ntok
            ((pg + a.kl_beta * klt).mean() / len(prompts)).backward()
            kls.append(klt.mean().item())
        if a.rl_replay > 0:
            (a.rl_replay_weight * replay_loss(ctx, a.rl_replay, a.replay_len)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        print(f"[rl r{rnd}] step {step+1}/{steps} reward {np.mean(rs[-a.rl_prompts*a.group:]):.3f} "
              f"kl {np.mean(kls[-a.rl_prompts:]):.4f}", flush=True)
    model.eval()
    return round(float(np.mean(rs)), 4), round(float(np.mean(kls)), 5)


def cmd_train(a):
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    a.test_files = a.test_file
    ctx = Ctx(a)
    MODE[0] = a.mode
    model, cfg, _ = load_model(a.ckpt); model_ref[0] = model
    ctx.log(stage="config", args=vars(a))
    t_start = time.time()
    evaluate(ctx, model, "base", 0)
    for f in a.extra_train.split(",") if a.extra_train else []:
        ctx.train += loop.lines(HERE / f)
    ctx.by_task = {}
    for r in ctx.train: ctx.by_task.setdefault(r["task"], []).append(r)
    register_phi(ctx.train + ctx.test)
    pool = [gold_segs(ctx.tok, r["prompt"]) for r in ctx.train]
    sft(ctx, model, pool, a.sft_steps, a.sft_lr, "sft")
    evaluate(ctx, model, "sft", 0); loop.save(ctx, model, cfg, "sft.pt")
    for rnd in range(1, a.ei_rounds + 1):
        kept = n = 0
        for r in ctx.rng.sample(ctx.train, a.ei_prompts):
            for _ in range(a.group):
                segs, info = rollout(model, ctx.tok, r["prompt"], a.temp); n += 1
                if info["reward"] == 1.0:
                    pool.append(segs); kept += 1
        sft(ctx, model, pool, a.ei_steps, a.ei_lr, f"ei{rnd}")
        evaluate(ctx, model, "expert_iteration", rnd, sampled=n, verified_kept=kept)
    loop.save(ctx, model, cfg, "ei.pt")
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    for rnd in range(1, a.rl_rounds + 1):
        if time.time() - t_start > a.max_hours * 3600:
            ctx.log(stage="stop", reason="max_hours", round=rnd); break
        rmean, kl = grpo_round(ctx, model, ref, a.rl_steps, rnd)
        evaluate(ctx, model, "grpo", rnd, reward_mean=rmean, kl_to_ref=kl)
        loop.save(ctx, model, cfg, f"rl-r{rnd}.pt")


# ---------------------------------------------------------------- llama.cpp
def llama_continue(binp, gguf, text, n):
    out = subprocess.run([f"{binp}/llama-simple", "-m", gguf, "-n", str(n), text],
                         capture_output=True, text=True).stdout
    return out[len(text):] if out.startswith(text) else None


def llama_rollout(binp, gguf, prompt, max_call=40, max_final=24):
    """Same tool loop, driving llama-simple (greedy). Returns (trajectory text, final)."""
    c = llama_continue(binp, gguf, prompt, max_call)
    if c is None: return None, None
    c = c.split("\n")[0]
    if "</call>" not in c: return prompt + c, None
    c = c[: c.index("</call>") + len("</call>")]
    m = CALL_RE.match(c)
    res = run_tool(m.group(1), m.group(2)) if m else None
    if res is None: return prompt + c, None
    text = prompt + c + f" <result>{res}</result>"
    f = llama_continue(binp, gguf, text, max_final)
    f = (f or "").split("\n")[0]
    return text + f, f


def cmd_llama(a):
    from tokenizers import Tokenizer
    a.n = a.n or 20
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(HERE.parent / "tokenizer/tokenizer.json"))
    model, _, _ = load_model(a.ckpt)
    rows = [json.loads(l) for l in open(HERE / a.test_file)]
    by = {}
    for r in rows: by.setdefault(r["task"], []).append(r)
    pick = [r for v in by.values() for r in v[: a.n // len(by)]]
    same = tool_used = 0
    for r in pick:
        segs, info = rollout(model, tok, r["prompt"])
        pt_text = tok.decode([t for s, _ in segs for t in s]).split("\n")[0]
        ll_text, ll_final = llama_rollout(a.llama_bin, a.gguf, r["prompt"])
        tool_used += info["wellformed"]
        ok = ll_text == pt_text
        same += ok
        if not ok: print("DIFF", repr(pt_text), "|", repr(ll_text))
    print(json.dumps({"prompts": len(pick), "tool_calls_executed_pytorch": tool_used,
                      "identical_trajectories": same}))


def cmd_eval(a):
    MODE[0] = a.mode
    from tokenizers import Tokenizer
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(HERE.parent / "tokenizer/tokenizer.json"))
    model, _, _ = load_model(a.ckpt)
    rows = [json.loads(l) for l in open(HERE / a.test_file)]
    register_phi(rows)
    if a.dump:  # per-prompt greedy trajectories, for parity checks against llama.cpp
        rows = rows[: a.n] if a.n else rows
        with open(a.dump, "w") as f:
            for r in rows:
                segs, info = rollout(model, tok, r["prompt"])
                f.write(json.dumps({"prompt": r["prompt"], "text": tok.decode([t for s, _ in segs for t in s]).split("\n")[0],
                                    "final": info.get("final"), "reward": info["reward"]}) + "\n")
        return
    print(json.dumps(evaluate_rows(model, tok, rows)))


def cmd_nodecheck(a):
    """isdss and card tool results vs greedy.mjs on every test-set call."""
    sets, cards = [], []
    for f in a.files.split(","):
        for l in open(HERE / f):
            r = json.loads(l); name, args = expected_call(r["prompt"])
            if name == "isdss": sets.append([int(x) for x in args.split()])
            if name == "card":
                xs = args.split(); i = xs.index("sum"); cards.append(([int(x) for x in xs[:i]], int(xs[i + 1])))
    js = (f"const g = await import('file://{a.kernel}');"
          "const d = JSON.parse(process.argv[1]);"
          "console.log(JSON.stringify({dss: d.sets.map(s => g.isDSS(s)),"
          "card: d.cards.map(([w, s]) => g.cardinalityFromSum(w, s))}));")
    ref = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", js,
                                              json.dumps({"sets": sets, "cards": cards})]))
    ok_d = sum((run_tool("isdss", " ".join(map(str, s))) == "yes") == r for s, r in zip(sets, ref["dss"]))
    ok_c = sum(run_tool("card", f"{' '.join(map(str, w))} sum {s}") == str(r) for (w, s), r in zip(cards, ref["card"]))
    print(json.dumps({"isdss_agree": ok_d, "isdss_total": len(sets), "card_agree": ok_c, "card_total": len(cards)}))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    for k, v in dict(threads=8, seed=0, batch=16, replay=4, replay_len=128, sft_steps=500, ei_rounds=1,
                     ei_prompts=200, ei_steps=150, rl_rounds=8, rl_steps=30, rl_prompts=8, group=8,
                     rl_replay=4, val_windows=64).items():
        t.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    for k, v in dict(replay_weight=1.0, sft_lr=2e-4, ei_lr=1e-4, rl_lr=2e-5, kl_beta=0.05, temp=1.0,
                     rl_replay_weight=0.5, max_hours=2.5).items():
        t.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    t.add_argument("--ckpt", required=True); t.add_argument("--out", required=True)
    t.add_argument("--train-file", default="data/train_v2.jsonl")
    t.add_argument("--test-file", default="data/test_balanced.jsonl")
    t.add_argument("--curriculum", action="store_true")
    t.add_argument("--mode", choices=["args", "intent"], default="args")
    t.add_argument("--extra-train", default="", help="comma-separated extra train jsonl files")
    for name in ("eval", "llama"):
        e = sub.add_parser(name); e.add_argument("ckpt"); e.add_argument("--threads", type=int, default=8)
        e.add_argument("--mode", choices=["args", "intent"], default="args")
        e.add_argument("--test-file", default="data/test_balanced.jsonl")
        e.add_argument("--dump", default=None); e.add_argument("--n", type=int, default=0)
        if name == "llama":
            e.add_argument("gguf")
            e.add_argument("--llama-bin", default="/workspace/llama.cpp/build/bin")
    nc = sub.add_parser("nodecheck")
    nc.add_argument("--files", default="data/test_balanced.jsonl,data/test.jsonl")
    nc.add_argument("--kernel", default="/workspace/sovereign-agi-kernel/runtime/greedy.mjs")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "llama": cmd_llama, "nodecheck": cmd_nodecheck}[a.cmd](a)


if __name__ == "__main__":
    main()
