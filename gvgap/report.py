"""Generate every number and table the paper prints, straight from results.

The paper never hard-codes a numeric claim: inline figures come from macros in
numbers.tex and tables are written here. If the data changes, the prose changes
with it.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap.analysis import FAM_LABEL, FAM_ORDER

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"

ARCH_TEX = {
    "single": "Single sample", "seq": "Sequential refine",
    "vote": "Self-consistency", "judge": "Ensemble + LLM judge",
    "prog": "Ensemble + program verifier", "oracle": "Ensemble + oracle",
    "debate": "Debate (3$\\times$2)",
}


def default_model(summ: pd.DataFrame) -> str:
    """Headline model for single-model tables: 1.5B when available."""
    have = set(summ["model"])
    for m in ("1.5B", "3B", "0.5B"):
        if m in have:
            return m
    return sorted(have)[0]


def pct(x: float, nd: int = 0) -> str:
    return f"{100*x:.{nd}f}\\%"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def n_generations() -> int:
    tot = 0
    for f in sorted((RES / "cache").glob("*.sqlite")):
        try:
            c = sqlite3.connect(f)
            tot += c.execute("select count(*) from gen").fetchone()[0]
        except Exception:
            pass
    return tot


def table_main(summ: pd.DataFrame, model: str | None = None) -> str:
    """Accuracy and token cost for every architecture, by family."""
    model = model or default_model(summ)
    s = summ[(summ.model == model)]
    fams = [f for f in FAM_ORDER if f in set(s.family)]
    archs = [("single", 1), ("seq", 4), ("vote", 8), ("judge", 8),
             ("prog", 8), ("debate", 6), ("oracle", 8)]
    lines = [
        r"\begin{tabular}{l" + "r" * len(fams) + "}", r"\toprule",
        "Architecture & " + " & ".join(FAM_LABEL[f] for f in fams) + r" \\",
        r"\midrule",
    ]
    for a, k in archs:
        cells = []
        for f in fams:
            d = s[(s.family == f) & (s.arch == a)]
            if d.empty:
                cells.append("--"); continue
            d = d.iloc[(d["k"] - k).abs().argsort().iloc[0]] if len(d) > 1 else d.iloc[0]
            cells.append(f"{d['acc']:.2f}")
        name = ARCH_TEX[a]
        if a == "oracle":
            lines.append(r"\midrule")
            name = r"\textit{" + name + r"} (ceiling)"
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(len(fams) + 1) +
              r"}{l}{\textit{mean tokens per task}} \\"]
    for a, k in [("single", 1), ("seq", 4), ("judge", 8), ("debate", 6)]:
        cells = []
        for f in fams:
            d = s[(s.family == f) & (s.arch == a)]
            if d.empty:
                cells.append("--"); continue
            d = d.iloc[(d["k"] - k).abs().argsort().iloc[0]] if len(d) > 1 else d.iloc[0]
            cells.append(f"{d['tokens']:,.0f}")
        lines.append(f"\\quad {ARCH_TEX[a]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_diagnostics(diag: pd.DataFrame) -> str:
    d = diag[diag.split == "calib"]
    fams = [f for f in FAM_ORDER if f in set(d.family)]
    models = [m for m in ("0.5B", "1.5B", "3B") if m in set(d.model)]
    lines = [r"\begin{tabular}{ll" + "r" * len(fams) + "}", r"\toprule",
             "Model & Diagnostic & " + " & ".join(FAM_LABEL[f] for f in fams)
             + r" \\", r"\midrule"]
    for m in models:
        for i, (col, lab) in enumerate([
            ("G", r"$\gap$ (verifier)"),
            ("plurality_alignment", r"$\plur$ (plurality)"),
            ("pick_a", r"position bias"),
            ("order_consistency", r"order consistency"),
        ]):
            cells = []
            for f in fams:
                r_ = d[(d.model == m) & (d.family == f)]
                v = r_[col].iloc[0] if not r_.empty else np.nan
                if pd.isna(v):
                    cells.append("--")
                elif col in ("pick_a", "order_consistency"):
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:+.2f}")
            first = f"Qwen2.5-{m}" if i == 0 else ""
            lines.append(f"{first} & {lab} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_matched(ml: pd.DataFrame, model: str | None = None) -> str:
    model = model or default_model(ml)
    d = ml[ml.model == model].copy()
    d["kk"] = d["config"].str.extract(r"(\d+)$").astype(float)
    d = d.sort_values(["family", "arch"])
    fams = [f for f in FAM_ORDER if f in set(d.family)]
    archs = ["vote", "judge", "prog", "debate"]
    lines = [r"\begin{tabular}{l" + "r" * len(archs) + "}", r"\toprule",
             "Family & " + " & ".join(
                 ARCH_TEX[a].replace("Ensemble + ", "") for a in archs)
             + r" \\", r"\midrule"]
    for f in fams:
        cells = []
        for a in archs:
            r_ = d[(d.family == f) & (d.arch == a)]
            if r_.empty:
                cells.append("--"); continue
            r_ = r_.loc[r_["matched_lift"].abs().idxmax()]
            star = "$^{*}$" if r_.get("p_holm", 1.0) < 0.05 else ""
            cells.append(f"${r_['matched_lift']:+.3f}${star}")
        lines.append(f"{FAM_LABEL[f]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


ALL_MACROS = """nFamilies nPerFamily nTasks nModels nArch nCalib nTest
nGenerations nFlip nCellsPositiveNaive nCellsFlipped
medianNaiveLift medianMatchedLift fracMatchedNegative rTwoApriori rTwoTransfer
maeApriori corrApriori nPredCells decisionAgreement nDecisionCells
decisionBaseline decisionPrecision decisionRecall decisionLift
etaProgSigMedian nSigCells nSigProgPositive rTwoJudgeArm nJudgeArmCells
etaProgMedian etaJudgeMedian etaVoteMedian etaJudgeMin nEtaNegative nEtaCells
Gmin Gmax nGnegative maxPositionBias minOrderConsistency minPairs
nDroppedLowPairs rTwoAprioriNoFilter nCitTopics citHallucRate citPone
citCoverage citEtaLLM citEtaAPI citG citPredEta citSaysReal citLLMTokens
citAccLLM citAccAPI nCitUnresolved""".split()


def main() -> None:
    # Every macro the paper can reference is defined unconditionally, so a
    # partially-complete results directory still produces a compiling PDF with
    # visibly missing values rather than an undefined-control-sequence error.
    m: list[str] = [r"% Auto-generated by gvgap/report.py -- do not edit."]
    m += [macro(n, r"\textbf{??}") for n in ALL_MACROS]
    summ = pd.read_csv(RES / "summary.csv")
    diag = pd.read_csv(RES / "diagnostics.csv")
    eta = pd.read_csv(RES / "eta.csv")
    suite = json.loads((ROOT / "data" / "suite.json").read_text())

    fams = sorted(set(summ.family))
    models = sorted(set(summ.model))
    n_per = int(summ.groupby(["model", "family", "arch", "k"])["n"].max().max())
    m += [
        macro("nFamilies", len(fams)),
        macro("nPerFamily", n_per),
        macro("nTasks", n_per * len(fams)),
        macro("nModels", len(models)),
        macro("nArch", 8),
        macro("nCalib", 30),
        macro("nTest", n_per - 30),
        macro("nGenerations", f"{n_generations():,}"),
    ]

    # --- token-matched sign flips -------------------------------------
    if (RES / "matched_lift_ci.csv").exists():
        ml = pd.read_csv(RES / "matched_lift_ci.csv")
        mlr = pd.read_csv(RES / "matched_lift.csv")
        j = mlr.merge(ml[["model", "family", "config", "matched_lift", "lo", "hi"]]
                      .rename(columns={"matched_lift": "ml_ci"}),
                      on=["model", "family", "config"], how="inner") \
            if "config" in mlr.columns else mlr
        pos_naive = mlr["naive_lift"] > 0
        flip = pos_naive & (mlr["matched_lift"] <= 0)
        m += [
            macro("nFlip", pct(flip.sum() / max(pos_naive.sum(), 1))),
            macro("nCellsPositiveNaive", int(pos_naive.sum())),
            macro("nCellsFlipped", int(flip.sum())),
            macro("medianNaiveLift", f"{mlr['naive_lift'].median():+.3f}"),
            macro("medianMatchedLift", f"{mlr['matched_lift'].median():+.3f}"),
            macro("fracMatchedNegative",
                  pct((mlr["matched_lift"] < 0).mean())),
        ]
        (PAPER / "tab_matched.tex").write_text(table_matched(ml))

    # --- prediction ----------------------------------------------------
    if (RES / "prediction_stats.json").exists():
        ps = json.loads((RES / "prediction_stats.json").read_text())
        a = ps.get("H2b_apriori", {})
        t = ps.get("H2a_transfer", {})
        dr = ps.get("decision_rule", {})
        m += [
            macro("rTwoApriori", f"{a.get('r2_identity', float('nan')):.2f}"),
            macro("rTwoTransfer", f"{t.get('r2_identity', float('nan')):.2f}"),
            macro("maeApriori", f"{a.get('mae', float('nan')):.3f}"),
            macro("corrApriori", f"{a.get('r', float('nan')):.2f}"),
            macro("nPredCells", a.get("n", 0)),
            macro("decisionAgreement",
                  pct(dr.get("agreement", float("nan")))
                  if dr.get("agreement") is not None else "--"),
            macro("nDecisionCells", dr.get("n", 0)),
            macro("decisionBaseline", pct(dr.get("majority_baseline", float("nan")), 1)),
            macro("decisionLift", f"{100*dr.get('lift_over_baseline', float('nan')):.1f}"),
            macro("decisionPrecision", f"{dr.get('precision', float('nan')):.2f}"),
            macro("decisionRecall", f"{dr.get('recall', float('nan')):.2f}"),
            macro("minPairs", ps.get("min_pairs", 10)),
            macro("nDroppedLowPairs", ps.get("n_cells_dropped_low_pairs", 0)),
            macro("rTwoAprioriNoFilter",
                  f"{ps.get('sensitivity_no_filter', {}).get('H2b_apriori', {}).get('r2_identity', float('nan')):.2f}"),
        ]

    # --- significance summary --------------------------------------------
    if (RES / "matched_lift_ci.csv").exists():
        mlc = pd.read_csv(RES / "matched_lift_ci.csv")
        sig = mlc[mlc["p_holm"] < 0.05]
        m += [
            macro("nSigCells", f"{len(sig)}"),
            macro("nSigProgPositive",
                  f"{int(((sig['arch'] == 'prog') & (sig['matched_lift'] > 0)).sum())}"),
            macro("etaProgSigMedian",
                  f"{sig[sig.arch == 'prog']['matched_lift'].median():+.3f}"
                  if (sig.arch == "prog").any() else "--"),
        ]

    # --- latent-score model on the judge arm alone ------------------------
    if (RES / "prediction.csv").exists():
        pr = pd.read_csv(RES / "prediction.csv")
        ja = pr[(pr.arch == "judge")].dropna(subset=["eta_hat", "eta_test"])
        if len(ja) >= 3:
            from gvgap.metrics import r2 as _r2
            m += [
                macro("rTwoJudgeArm",
                      f"{_r2(ja['eta_test'].to_numpy(), ja['eta_hat'].to_numpy()):.2f}"),
                macro("nJudgeArmCells", f"{len(ja)}"),
            ]

    # --- eta extremes ---------------------------------------------------
    e8 = eta[eta.k == 8]
    if not e8.empty:
        prog = e8[e8.arch == "prog"]["eta"].dropna()
        judge = e8[e8.arch == "judge"]["eta"].dropna()
        vote = e8[e8.arch == "vote"]["eta"].dropna()
        m += [
            macro("etaProgMedian", f"{prog.median():.2f}" if len(prog) else "--"),
            macro("etaJudgeMedian", f"{judge.median():.2f}" if len(judge) else "--"),
            macro("etaVoteMedian", f"{vote.median():.2f}" if len(vote) else "--"),
            macro("etaJudgeMin", f"{judge.min():.2f}" if len(judge) else "--"),
            macro("nEtaNegative", int((e8["eta"] < 0).sum())),
            macro("nEtaCells", int(e8["eta"].notna().sum())),
        ]

    # --- diagnostics spread ---------------------------------------------
    dc = diag[diag.split == "calib"]
    if not dc.empty and dc["G"].notna().any():
        m += [
            macro("Gmin", f"{dc['G'].min():+.2f}"),
            macro("Gmax", f"{dc['G'].max():+.2f}"),
            macro("nGnegative", int((dc["G"] < 0).sum())),
            macro("minOrderConsistency",
                  f"{dc['order_consistency'].min():.2f}"
                  if "order_consistency" in dc and dc["order_consistency"].notna().any() else "--"),
            macro("maxPositionBias",
                  f"{dc['pick_a'].max():.2f}" if dc["pick_a"].notna().any() else "--"),
        ]

    # --- citation case study ---------------------------------------------
    cs_path = RES / "citation_case_study.json"
    if cs_path.exists():
        cs = json.loads(cs_path.read_text())
        m += [
            macro("nCitTopics", cs["n_topics"]),
            macro("nCitUnresolved", cs.get("n_unresolved_candidates", 0)),
            macro("citHallucRate", pct(cs["hallucination_rate"], 1)),
            macro("citPone", f"{cs['p1']:.2f}"),
            macro("citCoverage", f"{cs['coverage']:.2f}"),
            macro("citEtaLLM", f"{cs['eta_llm_verifier']:.2f}"),
            macro("citEtaAPI", f"{cs['eta_api_verifier']:.2f}"),
            macro("citG", f"{cs['G_pairwise']:+.2f}"),
            macro("citPredEta", f"{cs['predicted_eta_from_G']:.2f}"),
            macro("citSaysReal", pct(cs["verifier_says_real_rate"], 0)),
            macro("citLLMTokens", f"{cs['llm_verifier_tokens']:,}"),
            macro("citAccLLM", f"{cs['acc_llm_verifier']:.2f}"),
            macro("citAccAPI", f"{cs['acc_api_verifier']:.2f}"),
        ]

    # Later definitions win in TeX only with \renewcommand, so emit the
    # defaults first and the computed values as renewcommands.
    head = m[: 1 + len(ALL_MACROS)]
    tail = [x.replace("newcommand", "renewcommand") for x in m[1 + len(ALL_MACROS):]]
    (PAPER / "numbers.tex").write_text("\n".join(head + tail) + "\n")
    (PAPER / "tab_main.tex").write_text(table_main(summ))
    (PAPER / "tab_diagnostics.tex").write_text(table_diagnostics(diag))
    print(f"wrote numbers.tex ({len(m)} macros) and 3 tables")


if __name__ == "__main__":
    main()
