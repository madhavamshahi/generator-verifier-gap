"""Main experiment driver.

For each (model, family) we build one pool of n=8 independent samples and then
evaluate every selector against that same pool, so all selector comparisons are
paired at the task level and share a generator. Sequential-refinement and
debate arms are generated separately because they change what the generator
sees.

Everything is written to results/raw/<model>/<family>.json and every generation
is cached, so the script is safe to interrupt and re-run.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap import arch
from gvgap.engine import Runner

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

FAMILIES = ["gsm8k", "csp", "logic", "arc", "wordcon", "mbpp"]
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # 3B omitted: see Limitations. Re-add to reproduce the third scale point.
    # "Qwen/Qwen2.5-3B-Instruct",
]
POOL_N = 8
K_VALUES = [4, 8]
SUBSETS = 2          # random sub-pools averaged over, for k < POOL_N
SEQ_ROUNDS = [1, 2, 3, 4]
LONG_MULTS = [2, 4]   # long-CoT single-agent budgets
DEBATE_CFGS = [(3, 2)]           # (agents, rounds)

MAX_TOKENS = {
    "gsm8k": 320, "csp": 320, "logic": 288,
    "arc": 256, "wordcon": 160, "mbpp": 384,
}


def load_items(fam: str) -> list[dict[str, Any]]:
    items = [json.loads(l) for l in (ROOT / "data" / f"{fam}.jsonl").open()]
    limit = int(os.environ.get("GVGAP_LIMIT", "0"))
    return items[:limit] if limit else items


def subsets_for(k: int, rng: random.Random) -> list[list[int]]:
    """Index subsets of the pool used to evaluate a k-agent system."""
    if k >= POOL_N:
        return [list(range(POOL_N))]
    return [rng.sample(range(POOL_N), k) for _ in range(SUBSETS)]


def run_family(runner: Runner, fam: str, log) -> dict[str, Any]:
    items = load_items(fam)
    mt = MAX_TOKENS[fam]
    t0 = time.time()
    pools = arch.build_pool(runner, items, POOL_N, max_tokens=mt)
    correct = arch.grade_pool(pools, items)
    log(f"  pool+grade {time.time() - t0:5.1f}s  p1={sum(c[0] for c in correct)/len(correct):.3f} "
        f"pass@8={sum(any(c) for c in correct)/len(correct):.3f}")

    rng = random.Random(7)
    per_task: list[dict[str, Any]] = [
        {
            "id": it["id"],
            "family": fam,
            "split": it["split"],
            "pool_correct": [bool(x) for x in cor],
            "pool_tokens": [g.total_tokens for g in pool],
            "pool_completion_tokens": [g.completion_tokens for g in pool],
            "arch": {},
        }
        for it, pool, cor in zip(items, pools, correct)
    ]

    # ---- selectors over the shared pool -------------------------------
    for k in K_VALUES:
        subs = subsets_for(k, rng)
        for name, fn in [
            ("vote", arch.select_majority),
            ("prog", arch.select_programmatic),
            ("oracle", arch.select_oracle),
        ]:
            for t, (it, pool, cor) in enumerate(zip(items, pools, correct)):
                accs, toks, defined = [], [], 0
                for sub in subs:
                    sp = [pool[i] for i in sub]
                    sc = [cor[i] for i in sub]
                    idx = fn(sp, sc, it, k)
                    cost = sum(g.total_tokens for g in sp)
                    if idx is None:
                        continue
                    defined += 1
                    accs.append(float(sc[idx]))
                    toks.append(cost)
                per_task[t]["arch"][f"{name}_k{k}"] = (
                    {
                        "acc": sum(accs) / len(accs),
                        "tokens": sum(toks) / len(toks),
                        "n_sub": defined,
                    }
                    if defined
                    else None
                )

        # ---- LLM judge (needs generation, so it is handled separately) --
        for si, sub in enumerate(subs):
            sub_pools = [[pool[i] for i in sub] for pool in pools]
            picks, gens, _ = arch.judge_select(
                runner, items, sub_pools, k, seed=100 + si
            )
            for t, (pick, g) in enumerate(zip(picks, gens)):
                key = f"judge_k{k}"
                slot = per_task[t]["arch"].setdefault(key, {"acc": [], "tokens": []})
                base = sum(x.total_tokens for x in sub_pools[t])
                sc = [correct[t][i] for i in sub]
                # judge_select indexes into the sub-pool it was handed, not
                # the full pool. An unparseable verdict falls back to the first
                # candidate, which is what a deployed system would have to do.
                j = pick if pick is not None and 0 <= pick < k else 0
                slot["acc"].append(float(sc[j]))
                slot["tokens"].append(base + g.total_tokens)
        for t in range(len(items)):
            key = f"judge_k{k}"
            slot = per_task[t]["arch"][key]
            per_task[t]["arch"][key] = {
                "acc": sum(slot["acc"]) / len(slot["acc"]),
                "tokens": sum(slot["tokens"]) / len(slot["tokens"]),
                "n_sub": len(slot["acc"]),
            }
        log(f"  k={k} selectors done ({time.time() - t0:5.1f}s)")

    # ---- single-agent sequential baseline ------------------------------
    for r in SEQ_ROUNDS:
        gens, budgets = arch.seq_refine(runner, items, rounds=r, max_tokens=mt)
        for t, (g, b) in enumerate(zip(gens, budgets)):
            from gvgap.verify import check

            per_task[t]["arch"][f"seq_r{r}"] = {
                "acc": float(check(g.text, items[t]).correct),
                "tokens": b.total_tokens,
                "n_sub": 1,
            }
    for mult in LONG_MULTS:
        gens, budgets = arch.seq_long(runner, items, mult, max_tokens=mt)
        for t, (g, b) in enumerate(zip(gens, budgets)):
            from gvgap.verify import check

            per_task[t]["arch"][f"long_m{mult}"] = {
                "acc": float(check(g.text, items[t]).correct),
                "tokens": b.total_tokens,
                "n_sub": 1,
            }
    log(f"  seq+long done ({time.time() - t0:5.1f}s)")

    # ---- debate ---------------------------------------------------------
    for agents, rounds in DEBATE_CFGS:
        final, budgets = arch.debate(runner, items, agents, rounds, max_tokens=mt)
        for t, (pool_f, b) in enumerate(zip(final, budgets)):
            cor_f = [arch.check(g.text, items[t]).correct for g in pool_f]
            idx = arch.select_majority(pool_f, cor_f, items[t], agents)
            acc = float(cor_f[idx]) if idx is not None else float(cor_f[0])
            per_task[t]["arch"][f"debate_a{agents}r{rounds}"] = {
                "acc": acc,
                "tokens": b.total_tokens,
                "n_sub": 1,
            }
    log(f"  debate done ({time.time() - t0:5.1f}s)")

    # ---- cheap diagnostics, computed per split -------------------------
    diags: dict[str, Any] = {}
    for split in ("calib", "test"):
        sel = [i for i, it in enumerate(items) if it["split"] == split]
        si, sp, sc = (
            [items[i] for i in sel],
            [pools[i] for i in sel],
            [correct[i] for i in sel],
        )
        diags[split] = {
            "gap": arch.pairwise_gap(runner, si, sp, sc, seed=11),
            "plurality": arch.plurality_alignment(si, sp, sc, POOL_N),
            "n": len(sel),
        }
    log(f"  diagnostics done ({time.time() - t0:5.1f}s)  "
        f"G_calib={diags['calib']['gap']['G']}")

    return {
        "model": runner.model_id,
        "family": fam,
        "pool_n": POOL_N,
        "max_tokens": mt,
        "tasks": per_task,
        "diagnostics": diags,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    models = sys.argv[1:] or MODELS
    for m in models:
        out_dir = RAW / m.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(m, batch_size=64)
        print(f"\n=== {m} ===", flush=True)
        for fam in FAMILIES:
            dest = out_dir / f"{fam}.json"
            if dest.exists():
                print(f"[{fam}] already done, skipping", flush=True)
                continue
            print(f"[{fam}]", flush=True)
            res = run_family(runner, fam, lambda s: print(s, flush=True))
            dest.write_text(json.dumps(res))
            print(f"  -> {dest}  ({res['wall_s']}s)", flush=True)
        print(runner.stats(), flush=True)


if __name__ == "__main__":
    main()
