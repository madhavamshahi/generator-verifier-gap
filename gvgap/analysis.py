"""Turn raw run records into the tables and quantities the paper reports."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap.metrics import (Est, boot, coverage_curve, holm, interp_accuracy,
                           paired_diff, pass_at_k, r2, selection_efficiency)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

SHORT = {
    "Qwen/Qwen2.5-0.5B-Instruct": "0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "3B",
}
FAM_ORDER = ["csp", "wordcon", "mbpp", "gsm8k", "arc", "logic"]
FAM_LABEL = {
    "csp": "CSP", "wordcon": "WordCon", "mbpp": "MBPP",
    "gsm8k": "GSM8K", "arc": "ARC", "logic": "Logic",
}
ARCH_LABEL = {
    "single": "Single", "vote": "Self-consistency", "judge": "Ensemble+judge",
    "prog": "Ensemble+prog. verifier", "oracle": "Ensemble+oracle",
    "seq": "Sequential refine", "debate": "Debate",
}


def load_raw() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RAW.glob("*/*.json"))]


def tidy(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (model, family, task, architecture)."""
    rows = []
    for rec in records:
        m = SHORT.get(rec["model"], rec["model"])
        fam = rec["family"]
        for t in rec["tasks"]:
            # The k=1 reference: the first pool sample, which every
            # architecture's first agent also draws.
            rows.append({
                "model": m, "family": fam, "task": t["id"], "split": t["split"],
                "arch": "single", "k": 1, "acc": float(t["pool_correct"][0]),
                "tokens": float(t["pool_tokens"][0]),
            })
            for name, v in t["arch"].items():
                if v is None:
                    continue
                mm = re.match(r"([a-z]+)_(?:k(\d+)|r(\d+)|m(\d+)|a(\d+)r(\d+))$", name)
                base = mm.group(1) if mm else name
                if base == "debate":
                    k = int(mm.group(5)) * int(mm.group(6))
                elif base == "seq":
                    k = int(mm.group(3))
                elif base == "long":
                    k = int(mm.group(4))
                else:
                    k = int(mm.group(2))
                rows.append({
                    "model": m, "family": fam, "task": t["id"], "split": t["split"],
                    "arch": base, "k": k, "acc": float(v["acc"]),
                    "tokens": float(v["tokens"]), "config": name,
                })
    return pd.DataFrame(rows)


def diagnostics(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for split, d in rec["diagnostics"].items():
            g, pl = d["gap"], d["plurality"]
            rows.append({
                "model": SHORT.get(rec["model"], rec["model"]),
                "family": rec["family"], "split": split,
                "G": g.get("G"), "gap_acc": g.get("acc"),
                "n_pairs": g.get("n_pairs"), "abstain": g.get("abstain_rate"),
                "pick_a": g.get("pick_a_rate"),
                "order_consistency": g.get("order_consistency"),
                "n_presentations": g.get("n_presentations"),
                "modal_share": pl.get("modal_share"),
                "plurality_alignment": pl.get("plurality_alignment"),
                "mode_accuracy": pl.get("mode_accuracy"),
            })
    return pd.DataFrame(rows)


def coverage_table(records: list[dict[str, Any]], kmax: int = 8) -> pd.DataFrame:
    rows = []
    for rec in records:
        cor = [t["pool_correct"] for t in rec["tasks"]]
        curve = coverage_curve(cor, kmax)
        for k, v in enumerate(curve, start=1):
            rows.append({
                "model": SHORT.get(rec["model"], rec["model"]),
                "family": rec["family"], "k": k, "pass_at_k": v,
            })
    return pd.DataFrame(rows)


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, family, arch, k): bootstrapped accuracy and mean token cost."""
    rows = []
    for (m, f, a, k), g in df.groupby(["model", "family", "arch", "k"]):
        e = boot(g["acc"].to_numpy())
        rows.append({
            "model": m, "family": f, "arch": a, "k": int(k),
            "acc": e.point, "lo": e.lo, "hi": e.hi,
            "tokens": float(g["tokens"].mean()), "n": len(g),
        })
    return pd.DataFrame(rows).sort_values(["model", "family", "arch", "k"])


def eta_table(summary: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Selection efficiency for every selector-based architecture."""
    rows = []
    for (m, f), g in summary.groupby(["model", "family"]):
        # pass@1 over the pool rather than the single measured draw: same
        # estimand, n times the data, and consistent with the latent-score
        # model's marginal (see predict.split_stats).
        c1 = cov[(cov.model == m) & (cov.family == f) & (cov.k == 1)]
        p1 = (float(c1["pass_at_k"].iloc[0]) if not c1.empty
              else float(g[(g.arch == "single")]["acc"].iloc[0]))
        for _, r in g.iterrows():
            if r["arch"] not in ("vote", "judge", "prog", "oracle"):
                continue
            c = cov[(cov.model == m) & (cov.family == f) & (cov.k == r["k"])]
            if c.empty:
                continue
            cval = float(c["pass_at_k"].iloc[0])
            rows.append({
                "model": m, "family": f, "arch": r["arch"], "k": int(r["k"]),
                "p1": p1, "coverage": cval, "headroom": cval - p1,
                "acc": r["acc"], "lift_vs_single": r["acc"] - p1,
                "eta": selection_efficiency(r["acc"], p1, cval),
                "tokens": r["tokens"],
            })
    return pd.DataFrame(rows)


def upper_envelope(tokens, acc):
    """Best accuracy achievable by a single agent at a budget or less.

    With two single-agent methods measured, the frontier is their upper
    envelope: sort by cost and take the running maximum. This is the
    conservative choice -- it never credits the baseline with a result we did
    not observe, and it never lets one weak method define the comparison.
    """
    t = np.asarray(tokens, dtype=float)
    a = np.asarray(acc, dtype=float)
    o = np.argsort(t)
    t, a = t[o], np.maximum.accumulate(a[o])
    return t, a


def matched_lift(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Net lift against a single agent priced at the same token budget.

    The single-agent frontier is the sequential-refinement curve: accuracy as a
    function of tokens spent inside one context. For each multi-agent cell we
    read that curve at the multi-agent cell's own token cost and take the
    difference. Comparing instead against a *single sample* -- as much of the
    literature does -- credits the multi-agent system for compute rather than
    for coordination.
    """
    rows = []
    for (m, f), g in summary.groupby(["model", "family"]):
        seq = g[g.arch.isin(["seq", "long"])].sort_values("tokens")
        if seq.empty:
            continue
        st, sa = seq["tokens"].to_numpy(), seq["acc"].to_numpy()
        # Anchor the low end of the frontier at the single-sample point.
        one = g[g.arch == "single"]
        if not one.empty:
            st = np.concatenate([[float(one["tokens"].iloc[0])], st])
            sa = np.concatenate([[float(one["acc"].iloc[0])], sa])
        st, sa = upper_envelope(st, sa)
        for _, r in g.iterrows():
            if r["arch"] in ("seq", "long", "single", "oracle"):
                continue
            base = interp_accuracy(st, sa, r["tokens"])
            rows.append({
                "model": m, "family": f, "arch": r["arch"], "k": int(r["k"]),
                "acc": r["acc"], "tokens": r["tokens"],
                "naive_lift": r["acc"] - float(one["acc"].iloc[0]),
                "matched_baseline": base,
                "matched_lift": r["acc"] - base,
            })
    return pd.DataFrame(rows)


def main() -> None:
    recs = load_raw()
    if not recs:
        print("no raw results yet")
        return
    out = ROOT / "results"
    df = tidy(recs)
    cov = coverage_table(recs)
    summ = cell_summary(df)
    diag = diagnostics(recs)
    eta = eta_table(summ, cov)
    ml = matched_lift(df, summ)
    for name, frame in [("tidy", df), ("coverage", cov), ("summary", summ),
                        ("diagnostics", diag), ("eta", eta), ("matched_lift", ml)]:
        frame.to_csv(out / f"{name}.csv", index=False)
    print(f"records={len(recs)}  rows={len(df)}")
    print(summ.groupby(["model", "family"]).size().to_string()[:600])


if __name__ == "__main__":
    main()


def matched_lift_ci(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 7
) -> pd.DataFrame:
    """Cluster bootstrap for the token-matched lift.

    Both arms are recomputed inside each resample -- the multi-agent cell's
    accuracy *and* the sequential frontier it is priced against -- because the
    baseline is itself an estimate. Treating the frontier as fixed would
    understate the interval.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (m, f), g in df.groupby(["model", "family"]):
        piv_acc = g.pivot_table(index="task", columns="config", values="acc")
        piv_tok = g.pivot_table(index="task", columns="config", values="tokens")
        single = g[g.arch == "single"].set_index("task")
        piv_acc["__single"] = single["acc"]
        piv_tok["__single"] = single["tokens"]
        seq_cols = [c for c in piv_acc.columns
                    if str(c).startswith(("seq_r", "long_m"))]
        if not seq_cols:
            continue
        targets = [
            c for c in piv_acc.columns
            if isinstance(c, str)
            and not c.startswith(("seq_r", "long_m", "oracle_", "__"))
        ]
        tasks = piv_acc.index.to_numpy()
        n = len(tasks)
        idx = rng.integers(0, n, size=(n_boot, n))

        acc_np = {c: piv_acc[c].to_numpy(dtype=float) for c in piv_acc.columns}
        tok_np = {c: piv_tok[c].to_numpy(dtype=float) for c in piv_tok.columns}

        def frontier(sel):
            t = [np.nanmean(tok_np["__single"][sel])] + [
                np.nanmean(tok_np[c][sel]) for c in seq_cols
            ]
            a = [np.nanmean(acc_np["__single"][sel])] + [
                np.nanmean(acc_np[c][sel]) for c in seq_cols
            ]
            return upper_envelope(t, a)

        full = np.arange(n)
        t0, a0 = frontier(full)
        for c in targets:
            point = float(np.nanmean(acc_np[c])) - interp_accuracy(
                t0, a0, float(np.nanmean(tok_np[c]))
            )
            draws = np.empty(n_boot)
            for b in range(n_boot):
                sel = idx[b]
                tb, ab = frontier(sel)
                draws[b] = float(np.nanmean(acc_np[c][sel])) - interp_accuracy(
                    tb, ab, float(np.nanmean(tok_np[c][sel]))
                )
            lo, hi = np.percentile(draws, [2.5, 97.5])
            centred = draws - draws.mean()
            p = 2.0 * min(
                float(np.mean(centred >= abs(point))),
                float(np.mean(centred <= -abs(point))),
            )
            rows.append({
                "model": m, "family": f, "config": c,
                "arch": c.split("_")[0],
                "matched_lift": point, "lo": float(lo), "hi": float(hi),
                "p": min(1.0, p),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].tolist())
    return out
