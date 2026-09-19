"""Smoke test: does every family produce a usable, non-degenerate signal?"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap import arch
from gvgap.engine import Runner

FAMS = ["gsm8k", "csp", "logic", "arc", "wordcon", "mbpp"]
ROOT = Path(__file__).resolve().parents[1]


def load(fam, n):
    items = [json.loads(l) for l in (ROOT / "data" / f"{fam}.jsonl").open()]
    return items[:n]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    k = 4
    r = Runner(model, batch_size=64)
    print(f"model={model} n={n} k={k}\n")
    print(f"{'family':8s} {'p1':>6s} {'pass@k':>7s} {'major':>7s} {'prog':>7s} {'G':>6s} {'tok/s':>7s}")
    for fam in FAMS:
        items = load(fam, n)
        t0 = time.time()
        pools = arch.build_pool(r, items, k, max_tokens=320)
        cor = arch.grade_pool(pools, items)
        p1 = sum(c[0] for c in cor) / len(cor)
        passk = sum(any(c) for c in cor) / len(cor)
        maj = [arch.select_majority(p, c, it, k) for p, c, it in zip(pools, cor, items)]
        maj_acc = (
            sum(cor[i][m] for i, m in enumerate(maj) if m is not None) / len(cor)
            if any(m is not None for m in maj) else float("nan")
        )
        prog = [arch.select_programmatic(p, c, it, k) for p, c, it in zip(pools, cor, items)]
        prog_acc = (
            sum(cor[i][m] for i, m in enumerate(prog) if m is not None) / len(cor)
            if any(m is not None for m in prog) else float("nan")
        )
        gap = arch.pairwise_gap(r, items, pools, cor)
        dt = time.time() - t0
        toks = sum(g.total_tokens for pool in pools for g in pool)
        print(f"{fam:8s} {p1:6.2f} {passk:7.2f} {maj_acc:7.2f} {prog_acc:7.2f} "
              f"{(gap['G'] if gap['G'] is not None else float('nan')):6.2f} {toks/dt:7.0f}")
    print("\n", r.stats())


if __name__ == "__main__":
    main()
