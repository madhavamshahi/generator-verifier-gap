"""H2: can cheap diagnostics measured on a small calibration set predict the
benefit of a multi-agent architecture on held-out tasks?

Two versions, in increasing order of practical value:

H2a (transfer). Selection efficiency eta measured on 30 calibration tasks
     predicts lift on 90 held-out tasks. This asks whether eta is a stable
     property rather than sampling noise.

H2b (a priori). The two cheap diagnostics -- pairwise verifier discrimination
     G and plurality alignment A -- predict held-out lift *without ever
     building the multi-agent system*. This is the version a practitioner can
     act on: measure on 30 labelled tasks, then decide whether to build.

Every feature is computed on the calibration split only; every target is
computed on the held-out split only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap.analysis import RAW, SHORT, load_raw
from gvgap.metrics import coverage_curve, interp_accuracy, pass_at_k, r2
from gvgap.theory import predict_eta

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
K = 8


def split_stats(rec: dict, split: str, k: int = K) -> dict:
    """p1, coverage and per-selector accuracy restricted to one split."""
    tasks = [t for t in rec["tasks"] if t["split"] == split]
    if not tasks:
        return {}
    cor = [t["pool_correct"] for t in tasks]
    # Single-sample accuracy is estimated as pass@1 over the whole pool
    # (equivalently the mean fraction correct), not from the first draw alone.
    # Both estimate the same quantity, but the pooled form uses n times as much
    # data and, crucially, is the same marginal the latent-score model assumes:
    # with it, a verifier at chance yields eta = 0 exactly rather than noise.
    p1 = float(np.mean([pass_at_k(len(c), int(sum(c)), 1) for c in cor]))
    cov = float(np.mean([pass_at_k(len(c), int(sum(c)), k) for c in cor]))
    out = {"p1": p1, "coverage": cov, "headroom": cov - p1, "n": len(tasks),
           "correct_counts": [int(sum(c)) for c in cor]}
    # The single-agent frontier restricted to this split: cost/accuracy of one
    # agent spending the same budget serially. Needed because the decision is
    # whether the ensemble beats *that*, not whether it beats one sample.
    single_tok = float(np.mean([t["pool_tokens"][0] for t in tasks]))
    fr = [(single_tok, p1)]
    for key in sorted({k for t in tasks for k in t["arch"]
                       if k.startswith(("seq_r", "long_m"))}):
        vals = [(t["arch"][key]["acc"], t["arch"][key]["tokens"])
                for t in tasks if t["arch"].get(key) is not None]
        if vals:
            fr.append((float(np.mean([v[1] for v in vals])),
                       float(np.mean([v[0] for v in vals]))))
    fr.sort()
    ft = [x[0] for x in fr]
    fa = list(np.maximum.accumulate([x[1] for x in fr]))  # upper envelope
    out["frontier_tokens"], out["frontier_acc"] = ft, fa
    for arch in ("vote", "judge", "prog", "debate"):
        key = f"{arch}_k{k}" if arch != "debate" else "debate_a3r2"
        vals = [t["arch"][key]["acc"] for t in tasks
                if t["arch"].get(key) is not None]
        toks = [t["arch"][key]["tokens"] for t in tasks
                if t["arch"].get(key) is not None]
        if vals:
            out[f"acc_{arch}"] = float(np.mean(vals))
            out[f"tok_{arch}"] = float(np.mean(toks))
            out[f"lift_{arch}"] = float(np.mean(vals)) - p1
            denom = cov - p1
            out[f"eta_{arch}"] = (
                (float(np.mean(vals)) - p1) / denom if abs(denom) > 1e-9 else np.nan
            )
    return out


def build() -> pd.DataFrame:
    recs = load_raw()
    rows = []
    for rec in recs:
        m, fam = SHORT.get(rec["model"], rec["model"]), rec["family"]
        cal, tst = split_stats(rec, "calib"), split_stats(rec, "test")
        if not cal or not tst:
            continue
        dc = rec["diagnostics"]["calib"]
        G = dc["gap"].get("G")
        A = dc["plurality"].get("plurality_alignment")
        # A discriminating pair needs one correct and one incorrect candidate
        # in the same pool. Where accuracy is very low or very high there are
        # few such pairs, so G is estimated from a handful of comparisons and
        # is correspondingly noisy. We record the count and pre-specify a
        # minimum for the headline analysis.
        # Distinct discriminating pairs, not presentations: counterbalancing
        # shows each pair twice, which doubles the evidence about the judge's
        # *ordering* behaviour but not the number of independent task items.
        n_pairs = dc["gap"].get("n_pairs") or 0
        pick_a = dc["gap"].get("pick_a_rate")
        for arch in ("vote", "judge", "prog"):
            if f"lift_{arch}" not in tst:
                continue
            # A priori predictions, from theory + calibration diagnostics only.
            if arch == "judge":
                # The latent-score model of Section 3 turns the *pairwise*
                # diagnostic G into a *listwise* selection prediction, using
                # only calibration-split quantities and no fitted parameters.
                if G is None:
                    eta_hat, pred_b = np.nan, np.nan
                else:
                    _, eta_hat = predict_eta(
                        cal["correct_counts"], K, float(G),
                        cal["p1"], cal["coverage"])
                    pred_b = eta_hat * cal["headroom"]
            elif arch == "vote":
                eta_hat = np.nan
                pred_b = np.nan if A is None else float(A)
            else:
                # A program that checks the task's own stated constraints is
                # assumed sound; its recall is what limits it, so the a priori
                # estimate is the full headroom.
                eta_hat = 1.0
                pred_b = cal["headroom"]
            # What one agent gains from the same budget, measured on
            # calibration only. Equation (rule) compares the predicted
            # ensemble gain against this, not against zero.
            tok = cal.get(f"tok_{arch}")
            seq_gain = (
                interp_accuracy(cal["frontier_tokens"], cal["frontier_acc"], tok)
                - cal["p1"]
            ) if tok else np.nan
            rows.append({
                "model": m, "family": fam, "arch": arch,
                "G_calib": G, "A_calib": A,
                "seq_gain_calib": seq_gain,
                "pred_matched": pred_b - seq_gain,
                "n_pairs": n_pairs, "pick_a": pick_a,
                # eta_hat and eta_test are both dimensionless selection
                # efficiencies, each computed entirely within its own split,
                # so they are directly comparable. Pairing a calibration-based
                # prediction with a full-set observation would manufacture
                # spread that is not there.
                "eta_hat": eta_hat,
                "eta_test": tst.get(f"eta_{arch}"),
                "headroom_calib": cal["headroom"],
                "headroom_test": tst["headroom"],
                "eta_calib": cal.get(f"eta_{arch}"),
                "pred_transfer": cal.get(f"eta_{arch}", np.nan) * cal["headroom"],
                "pred_apriori": pred_b,
                "actual": tst[f"lift_{arch}"],
                "p1_test": tst["p1"],
            })
    return pd.DataFrame(rows)


MIN_PAIRS = 10   # pre-specified; sensitivity to this choice is reported


def evaluate(df: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> dict:
    out = {"min_pairs": min_pairs}
    for name, col in (("H2a_transfer", "pred_transfer"),
                      ("H2b_apriori", "pred_apriori")):
        d = df.dropna(subset=[col, "actual"])
        # The pair-count filter applies only where the prediction depends on G;
        # self-consistency uses plurality alignment, which every task informs.
        d = d[(d["arch"] != "judge") | (d["n_pairs"] >= min_pairs)]
        if len(d) < 3:
            out[name] = {"n": len(d)}
            continue
        y, yh = d["actual"].to_numpy(), d[col].to_numpy()
        # Pearson r and the identity-line R^2. We report the identity-line R^2
        # because the theory predicts the *value* of the lift, not merely a
        # quantity correlated with it; a fitted slope would hide bias.
        r = float(np.corrcoef(y, yh)[0, 1])
        out[name] = {
            "n": int(len(d)),
            "r": r,
            "r2_identity": float(r2(y, yh)),
            "r2_pearson": float(r ** 2),
            "mae": float(np.mean(np.abs(y - yh))),
            "bias": float(np.mean(yh - y)),
        }
    return out


def decision_rule(df: pd.DataFrame) -> dict:
    """Does the a priori diagnostic call the build / don't-build decision right?

    The decision is over the *token-matched* lift, since that is what actually
    determines whether the architecture was worth building.
    """
    ml = pd.read_csv(RES / "matched_lift_ci.csv")
    ml["kk"] = ml["config"].str.extract(r"k(\d+)$").astype(float)
    ml = ml[(ml.kk == K) | ml.kk.isna()]
    rows = []
    for _, r in df.iterrows():
        sub = ml[(ml.model == r["model"]) & (ml.family == r["family"])
                 & (ml.arch == r["arch"])]
        if sub.empty or pd.isna(r.get("pred_matched")):
            continue
        actual = float(sub["matched_lift"].iloc[0])
        rows.append({
            "model": r["model"], "family": r["family"], "arch": r["arch"],
            "pred_apriori": r["pred_apriori"],
            "seq_gain_calib": r["seq_gain_calib"],
            "pred_matched": r["pred_matched"],
            "matched_lift": actual,
            "pred_build": bool(r["pred_matched"] > 0),
            "actually_helped": bool(actual > 0),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n": 0}
    agree = (d["pred_build"] == d["actually_helped"]).mean()
    d.to_csv(RES / "decision_rule.csv", index=False)
    tp = int((d["pred_build"] & d["actually_helped"]).sum())
    fp = int((d["pred_build"] & ~d["actually_helped"]).sum())
    fn = int((~d["pred_build"] & d["actually_helped"]).sum())
    tn = int((~d["pred_build"] & ~d["actually_helped"]).sum())
    helped = float(d["actually_helped"].mean())
    # The rule must be judged against always guessing the majority class,
    # which on an imbalanced set is a strong and often-ignored baseline.
    majority = max(helped, 1 - helped)
    return {
        "n": int(len(d)),
        "agreement": float(agree),
        "majority_baseline": float(majority),
        "lift_over_baseline": float(agree - majority),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_helped": int(d["actually_helped"].sum()),
        "n_predicted_build": int(d["pred_build"].sum()),
    }


def main() -> None:
    df = build()
    if df.empty:
        print("no data yet")
        return
    res = evaluate(df)
    # Sensitivity: the same analysis with no pair-count filter at all.
    res["sensitivity_no_filter"] = evaluate(df, min_pairs=0)
    res["n_cells_dropped_low_pairs"] = int(
        ((df["arch"] == "judge") & (df["n_pairs"] < MIN_PAIRS)).sum())
    for name, col in (("H2a_transfer", "pred_transfer"),
                      ("H2b_apriori", "pred_apriori")):
        d = df.dropna(subset=[col, "actual"]).copy()
        d = d[(d["arch"] != "judge") | (d["n_pairs"] >= MIN_PAIRS)]
        d["predicted"] = d[col]
        d["r2"] = res[name].get("r2_identity", np.nan)
        d.to_csv(RES / ("prediction.csv" if name == "H2b_apriori"
                        else "prediction_transfer.csv"), index=False)
    try:
        res["decision_rule"] = decision_rule(df)
    except FileNotFoundError:
        pass
    (RES / "prediction_stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
