"""Compact human-readable summary of every claim the paper makes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap.analysis import FAM_ORDER

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    summ = pd.read_csv(RES / "summary.csv")
    cov = pd.read_csv(RES / "coverage.csv")
    eta = pd.read_csv(RES / "eta.csv")
    diag = pd.read_csv(RES / "diagnostics.csv")
    models = [m for m in ("0.5B", "1.5B", "3B") if m in set(summ.model)]

    print("=" * 78)
    print("A. GENERATOR: single-sample accuracy and coverage headroom (k=8)")
    print("=" * 78)
    print(f"{'model':6s} {'family':8s} {'p1':>6s} {'pass@8':>7s} {'headroom':>9s}")
    for m in models:
        for f in FAM_ORDER:
            s = summ[(summ.model == m) & (summ.family == f) & (summ.arch == "single")]
            c = cov[(cov.model == m) & (cov.family == f) & (cov.k == 8)]
            if s.empty or c.empty:
                continue
            p1, c8 = float(s["acc"].iloc[0]), float(c["pass_at_k"].iloc[0])
            print(f"{m:6s} {f:8s} {p1:6.3f} {c8:7.3f} {c8-p1:+9.3f}")

    print()
    print("=" * 78)
    print("B. DIAGNOSTICS on calibration split (G = verifier gap, A = plurality)")
    print("=" * 78)
    d = diag[diag.split == "calib"]
    print(f"{'model':6s} {'family':8s} {'G':>7s} {'A':>7s} {'pickA':>7s} {'abstain':>8s} {'npair':>6s}")
    for m in models:
        for f in FAM_ORDER:
            r = d[(d.model == m) & (d.family == f)]
            if r.empty:
                continue
            r = r.iloc[0]
            g = "  n/a" if pd.isna(r["G"]) else f"{r['G']:+7.3f}"
            a = "  n/a" if pd.isna(r["plurality_alignment"]) else f"{r['plurality_alignment']:+7.3f}"
            pa = "  n/a" if pd.isna(r["pick_a"]) else f"{r['pick_a']:7.2f}"
            print(f"{m:6s} {f:8s} {g} {a} {pa} {r['abstain']:8.2f} {int(r['n_pairs']):6d}")

    print()
    print("=" * 78)
    print("C. SELECTION EFFICIENCY eta at k=8")
    print("=" * 78)
    e = eta[eta.k == 8]
    print(f"{'model':6s} {'family':8s} {'vote':>8s} {'judge':>8s} {'prog':>8s}")
    for m in models:
        for f in FAM_ORDER:
            row = [m, f]
            cells = []
            for a in ("vote", "judge", "prog"):
                r = e[(e.model == m) & (e.family == f) & (e.arch == a)]
                cells.append("     n/a" if r.empty or pd.isna(r["eta"].iloc[0])
                             else f"{r['eta'].iloc[0]:+8.3f}")
            if all(c.strip() == "n/a" for c in cells):
                continue
            print(f"{m:6s} {f:8s} " + " ".join(cells))

    print()
    print("=" * 78)
    print("D. TOKEN-MATCHED LIFT (vs single-agent frontier at equal tokens)")
    print("=" * 78)
    if (RES / "matched_lift_ci.csv").exists():
        ml = pd.read_csv(RES / "matched_lift_ci.csv")
        mlr = pd.read_csv(RES / "matched_lift.csv")
        print(f"{'model':6s} {'family':8s} {'config':14s} {'naive':>8s} {'matched':>8s} {'95% CI':>18s} {'p_holm':>8s}")
        for _, r in ml.sort_values(["model", "family", "config"]).iterrows():
            n = mlr[(mlr.model == r["model"]) & (mlr.family == r["family"])
                    & (mlr.arch == r["arch"])]
            nl = f"{n['naive_lift'].iloc[0]:+8.3f}" if not n.empty else "     n/a"
            ci = f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
            print(f"{r['model']:6s} {r['family']:8s} {r['config']:14s} {nl} "
                  f"{r['matched_lift']:+8.3f} {ci:>18s} {r.get('p_holm', float('nan')):8.3f}")
        pos = mlr["naive_lift"] > 0
        print(f"\n  cells with positive naive lift : {int(pos.sum())}/{len(mlr)}")
        print(f"  of those, non-positive matched : {int((pos & (mlr['matched_lift'] <= 0)).sum())}")
        print(f"  all cells with negative matched: {int((mlr['matched_lift'] < 0).sum())}/{len(mlr)}")
        print(f"  median naive  lift: {mlr['naive_lift'].median():+.3f}")
        print(f"  median matched lift: {mlr['matched_lift'].median():+.3f}")

    print()
    print("=" * 78)
    print("E. PREDICTION (H2)")
    print("=" * 78)
    if (RES / "prediction_stats.json").exists():
        print(json.dumps(json.loads((RES / "prediction_stats.json").read_text()), indent=1))

    print()
    print("=" * 78)
    print("F. ARCHITECTURE ACCURACY / COST, per model")
    print("=" * 78)
    for m in models:
        print(f"\n--- {m} ---")
        piv = summ[summ.model == m].pivot_table(
            index=["arch", "k"], columns="family", values="acc")
        cols = [c for c in FAM_ORDER if c in piv.columns]
        print(piv[cols].round(3).to_string())
        tok = summ[summ.model == m].pivot_table(
            index=["arch", "k"], columns="family", values="tokens")
        print("\ntokens:")
        print(tok[cols].round(0).to_string())


if __name__ == "__main__":
    main()
