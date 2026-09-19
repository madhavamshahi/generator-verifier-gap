"""Publication figures.

Style notes: a colourblind-safe categorical order held fixed across every
figure (a given architecture is the same hue everywhere), recessive grids,
no top/right spines, and direct labels wherever a legend would otherwise be
the only carrier of identity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap.analysis import FAM_LABEL, FAM_ORDER

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"

# Validated categorical order (see dataviz palette validation in the appendix).
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948",
}
ARCH_COLOR = {
    "single": "#6b6b6b", "seq": C["blue"], "long": C["yellow"],
    "vote": C["orange"],
    "judge": C["violet"], "prog": C["aqua"], "oracle": "#9a9a9a",
    "debate": C["magenta"],
}
ARCH_NAME = {
    "single": "Single sample", "seq": "Sequential refine",
    "long": "Long chain-of-thought",
    "vote": "Self-consistency", "judge": "Ensemble + LLM judge",
    "prog": "Ensemble + program verifier", "oracle": "Ensemble + oracle",
    "debate": "Debate",
}
MODEL_COLOR = {"0.5B": C["yellow"], "1.5B": C["blue"], "3B": C["violet"]}
INK, MUTED, GRID = "#111111", "#52514e", "#dcdcd8"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def default_model() -> str:
    """Headline model for single-model figures: 1.5B when available."""
    have = set(pd.read_csv(RES / "summary.csv")["model"])
    for m in ("1.5B", "3B", "0.5B"):
        if m in have:
            return m
    return sorted(have)[0]


def _read():
    return (
        pd.read_csv(RES / "summary.csv"),
        pd.read_csv(RES / "coverage.csv"),
        pd.read_csv(RES / "eta.csv"),
        pd.read_csv(RES / "diagnostics.csv"),
    )


# --------------------------------------------------------------- Figure 1
def fig_thesis(model: str | None = None, hi: str = "mbpp", lo: str = "logic") -> None:
    """The decomposition, shown on a high- and a low-asymmetry family.

    The shaded band is the coverage headroom: accuracy that *exists* in the
    candidate pool. Where a selector's curve sits inside that band is eta.
    """
    model = model or default_model()
    summ, cov, eta, _ = _read()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, fam in zip(axes, (hi, lo)):
        c = cov[(cov.model == model) & (cov.family == fam)].sort_values("k")
        s = summ[(summ.model == model) & (summ.family == fam)]
        if c.empty or s.empty:
            continue
        ks = c["k"].to_numpy()
        p1 = float(s[s.arch == "single"]["acc"].iloc[0])
        ax.fill_between(ks, p1, c["pass_at_k"], color=C["blue"], alpha=0.10,
                        lw=0, zorder=1)
        ax.plot(ks, c["pass_at_k"], color=INK, lw=1.6, zorder=4)
        ax.axhline(p1, color="#6b6b6b", lw=1.2, ls=(0, (4, 3)), zorder=3)
        # Collect end-of-line labels, then push them apart vertically so that
        # curves ending at similar accuracy do not overprint each other.
        labels = [(float(c["pass_at_k"].iloc[-1]), "coverage (pass@k)", INK)]
        for arch in ("vote", "judge", "prog"):
            d = s[s.arch == arch].sort_values("k")
            if d.empty:
                continue
            ax.plot(d["k"], d["acc"], color=ARCH_COLOR[arch], lw=2,
                    marker="o", ms=4.5, mew=1.2, mec="white", zorder=5)
            labels.append((float(d["acc"].iloc[-1]),
                           ARCH_NAME[arch].replace("Ensemble + ", ""),
                           ARCH_COLOR[arch]))
        labels.sort()
        min_gap = 0.052
        placed = []
        for y, text, colr in labels:
            if placed and y - placed[-1][0] < min_gap:
                y = placed[-1][0] + min_gap
            placed.append((y, text, colr))
        # A white halo keeps labels legible where they cross the reference
        # line or a curve, without moving them off the series they name.
        halo = [pe.withStroke(linewidth=2.6, foreground="white")]
        for (y, text, colr), (y_true, _, _) in zip(placed, labels):
            ax.annotate(text, (8, y_true), xytext=(8.55, y),
                        textcoords="data", color=colr, fontsize=7,
                        va="center", ha="left", zorder=8,
                        path_effects=halo,
                        arrowprops=dict(arrowstyle="-", color=colr,
                                        lw=0.6, shrinkA=2, shrinkB=0)
                        if abs(y - y_true) > 0.012 else None)
        ax.annotate("single sample", (1.05, p1), textcoords="offset points",
                    xytext=(0, -12), fontsize=7, color="#6b6b6b", ha="left",
                    zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        headroom = float(c["pass_at_k"].iloc[-1]) - p1
        ax.set_title(f"{FAM_LABEL[fam]}   (headroom {headroom:+.2f})")
        ax.set_xlabel("agents / samples $k$")
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.set_xlim(0.8, 14.2)
        despine(ax)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.0)
    fig.suptitle(
        f"Same generator, same headroom, different selectors  ({model})",
        fontsize=9.5, y=1.04)
    save(fig, "fig1_thesis")


# --------------------------------------------------------------- Figure 2
def fig_coverage() -> None:
    """pass@k per family and model: how much headroom sampling actually buys."""
    _, cov, _, _ = _read()
    fams = [f for f in FAM_ORDER if f in set(cov.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(9.5, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        for m in ("0.5B", "1.5B", "3B"):
            d = cov[(cov.model == m) & (cov.family == fam)].sort_values("k")
            if d.empty:
                continue
            ax.plot(d["k"], d["pass_at_k"], color=MODEL_COLOR[m], lw=1.8,
                    marker="o", ms=3, mew=0.8, mec="white")
        ax.set_title(FAM_LABEL[fam])
        ax.set_xticks([1, 4, 8])
        ax.set_xlabel("$k$")
        despine(ax)
    axes[0].set_ylabel("pass@$k$")
    axes[0].set_ylim(0, 1.0)
    handles = [Line2D([], [], color=MODEL_COLOR[m], lw=1.8, label=f"Qwen2.5-{m}")
               for m in ("0.5B", "1.5B", "3B")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig2_coverage")


# --------------------------------------------------------------- Figure 3
def fig_eta(k: int = 8) -> None:
    """Selection efficiency by family and selector."""
    _, _, eta, _ = _read()
    d = eta[eta.k == k]
    fams = [f for f in FAM_ORDER if f in set(d.family)]
    models = [m for m in ("0.5B", "1.5B", "3B") if m in set(d.model)]
    archs = ["vote", "judge", "prog"]
    fig, axes = plt.subplots(1, len(models), figsize=(9.5, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.26
    for ax, m in zip(axes, models):
        x = np.arange(len(fams))
        for j, a in enumerate(archs):
            vals = []
            for f in fams:
                r = d[(d.model == m) & (d.family == f) & (d.arch == a)]
                vals.append(float(r["eta"].iloc[0]) if not r.empty else np.nan)
            ax.bar(x + (j - 1) * w, vals, width=w - 0.03,
                   color=ARCH_COLOR[a], edgecolor="white", linewidth=0.8)
        ax.axhline(0, color=MUTED, lw=0.8)
        # The oracle ceiling, shown on every panel so the two are comparable.
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([FAM_LABEL[f] for f in fams], rotation=35,
                           ha="right", fontsize=7.5)
        ax.set_title(f"Qwen2.5-{m}")
        despine(ax)
    axes[0].set_ylabel(r"selection efficiency $\eta$")
    axes[0].annotate("oracle", (0.02, 1.0), xycoords=("axes fraction", "data"),
                     fontsize=6.5, color=MUTED, va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ARCH_COLOR[a],
                             label=ARCH_NAME[a].replace("Ensemble + ", ""))
               for a in archs]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig3_eta")


# --------------------------------------------------------------- Figure 4
def fig_frontier(model: str | None = None) -> None:
    """Accuracy against token cost, with the single-agent frontier drawn in.

    A multi-agent point above the line earns its compute; a point below it is
    a system that would have done better as one agent thinking longer.
    """
    model = model or default_model()
    summ, _, _, _ = _read()
    s = summ[summ.model == model]
    fams = [f for f in FAM_ORDER if f in set(s.family)]
    fig, axes = plt.subplots(1, len(fams), figsize=(11.0, 2.5))
    axes = np.atleast_1d(axes)
    # Each panel keeps its own y-scale (accuracy ranges differ by family), so
    # they need room for their own tick labels.
    fig.subplots_adjust(wspace=0.42)
    for ax, fam in zip(axes, fams):
        d = s[s.family == fam]
        one = d[d.arch == "single"]
        sing = d[d.arch.isin(["seq", "long"])].sort_values("tokens")
        if one.empty or sing.empty:
            continue
        t = np.concatenate([one["tokens"].to_numpy(), sing["tokens"].to_numpy()])
        a = np.concatenate([one["acc"].to_numpy(), sing["acc"].to_numpy()])
        o = np.argsort(t)
        t, env = t[o], np.maximum.accumulate(a[o])
        ax.plot(t, env, color=ARCH_COLOR["seq"], lw=1.8, zorder=3,
                drawstyle="steps-post")
        for arch2, mk2 in (("seq", "o"), ("long", "*")):
            dd2 = d[d.arch == arch2]
            if dd2.empty:
                continue
            ax.scatter(dd2["tokens"], dd2["acc"], s=22, marker=mk2,
                       color=ARCH_COLOR[arch2], edgecolor="white",
                       linewidth=0.7, zorder=4)
        for arch, mk in (("vote", "s"), ("judge", "^"), ("prog", "D"),
                         ("debate", "v")):
            dd = d[d.arch == arch]
            if dd.empty:
                continue
            ax.scatter(dd["tokens"], dd["acc"], s=26, marker=mk,
                       color=ARCH_COLOR[arch], edgecolor="white",
                       linewidth=0.8, zorder=5)
        ax.set_title(FAM_LABEL[fam])
        ax.set_xscale("log")
        ax.set_xlabel("tokens / task")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        despine(ax)
    axes[0].set_ylabel("accuracy")
    handles = [Line2D([], [], color=ARCH_COLOR["seq"], lw=1.8,
                      label="Single-agent frontier"),
               Line2D([], [], color=ARCH_COLOR["long"], marker="*", ls="",
                      ms=7, label="Long chain-of-thought")]
    handles += [Line2D([], [], color=ARCH_COLOR[a], marker=mk, ls="",
                       ms=5, label=ARCH_NAME[a])
                for a, mk in (("vote", "s"), ("judge", "^"), ("prog", "D"),
                              ("debate", "v"))]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.30))
    fig.suptitle(f"Cost-accuracy frontier, Qwen2.5-{model}", fontsize=9.5, y=1.06)
    save(fig, "fig4_frontier")


# --------------------------------------------------------------- Figure 5
def fig_matched_lift(model: str | None = None) -> None:
    """Token-matched lift with bootstrap intervals; the sign is the story."""
    model = model or default_model()
    ml = pd.read_csv(RES / "matched_lift_ci.csv")
    d = ml[ml.model == model]
    fams = [f for f in FAM_ORDER if f in set(d.family)]
    archs = ["vote", "judge", "prog", "debate"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(fams))
    w = 0.2
    for j, a in enumerate(archs):
        pts, los, his = [], [], []
        for f in fams:
            r = d[(d.family == f) & (d.arch == a)]
            if r.empty:
                pts.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            r = r.loc[r["matched_lift"].abs().idxmax()]
            pts.append(r["matched_lift"]); los.append(r["lo"]); his.append(r["hi"])
        pts = np.array(pts, float)
        err = np.vstack([pts - np.array(los, float), np.array(his, float) - pts])
        ax.bar(x + (j - 1.5) * w, pts, width=w - 0.03, color=ARCH_COLOR[a],
               edgecolor="white", linewidth=0.8,
               label=ARCH_NAME[a].replace("Ensemble + ", ""))
        ax.errorbar(x + (j - 1.5) * w, pts, yerr=np.abs(err), fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in fams])
    ax.set_ylabel("accuracy vs. token-matched single agent")
    ax.set_title(f"Token-matched lift, Qwen2.5-{model}", fontsize=9.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    despine(ax)
    save(fig, "fig5_matched_lift")


# --------------------------------------------------------------- Figure 6
def fig_prediction() -> None:
    """H2: do calibration-set diagnostics predict held-out lift?"""
    pred = pd.read_csv(RES / "prediction.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    lim = [
        min(pred["actual"].min(), pred["predicted"].min()) - 0.05,
        max(pred["actual"].max(), pred["predicted"].max()) + 0.05,
    ]
    ax.plot(lim, lim, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for a, g in pred.groupby("arch"):
        ax.scatter(g["predicted"], g["actual"], s=30,
                   color=ARCH_COLOR.get(a, C["blue"]), edgecolor="white",
                   linewidth=0.8, zorder=4,
                   label=ARCH_NAME.get(a, a).replace("Ensemble + ", ""))
    r2v = float(pred["r2"].iloc[0])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted lift (calibration diagnostics)")
    ax.set_ylabel("observed held-out lift")
    ax.set_title(f"Held-out prediction  ($R^2={r2v:.2f}$)", fontsize=9)
    ax.legend(loc="upper left")
    despine(ax)
    save(fig, "fig6_prediction")


# --------------------------------------------------------------- Figure 7
def fig_scale() -> None:
    """How the two diagnostics move with model size."""
    _, _, eta, diag = _read()
    d = diag[diag.split == "test"]
    fams = [f for f in FAM_ORDER if f in set(d.family)]
    order = ["0.5B", "1.5B", "3B"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (col, lab) in zip(axes, [
        ("G", "verifier discrimination $G$"),
        ("plurality_alignment", "plurality alignment $A$"),
    ]):
        for f in fams:
            ys, xs = [], []
            for i, m in enumerate(order):
                r = d[(d.model == m) & (d.family == f)]
                if r.empty or pd.isna(r[col].iloc[0]):
                    continue
                xs.append(i); ys.append(float(r[col].iloc[0]))
            if not xs:
                continue
            ax.plot(xs, ys, lw=1.6, marker="o", ms=4, mew=0.8, mec="white",
                    color=FAM_COLOR[f])
            ax.annotate(FAM_LABEL[f], (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(5, 0),
                        fontsize=7, color=FAM_COLOR[f], va="center")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"Qwen2.5-{m}" for m in order], fontsize=7.5)
        ax.set_ylabel(lab)
        ax.set_xlim(-0.2, 2.9)
        despine(ax)
    fig.suptitle("Both diagnostics are model-dependent, not task-intrinsic",
                 fontsize=9.5, y=1.04)
    save(fig, "fig7_scale")


FAM_COLOR = {
    "csp": C["aqua"], "wordcon": C["green"], "mbpp": C["blue"],
    "gsm8k": C["orange"], "arc": C["violet"], "logic": C["red"],
}


def main() -> None:
    style()
    for fn in (fig_thesis, fig_coverage, fig_eta, fig_frontier,
               fig_matched_lift, fig_prediction, fig_scale, fig_theory):
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {exc}")



# --------------------------------------------------------------- Figure 8
def fig_theory() -> None:
    """Does the latent-score model predict observed selection efficiency?

    Both axes are computed strictly within their own split: the prediction uses
    the calibration gap and calibration pool composition, the observation uses
    the held-out tasks. Mixing the two would silently manufacture spread.
    Nothing is fitted, so the reference is the identity line.
    """
    pred = pd.read_csv(RES / "prediction.csv")
    m = pred[(pred.arch == "judge")].dropna(subset=["eta_hat", "eta_test"])
    if m.empty:
        raise ValueError("no judge cells with both predicted and observed eta")
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    lim = [min(m["eta_hat"].min(), m["eta_test"].min()) - 0.08,
           max(m["eta_hat"].max(), m["eta_test"].max()) + 0.08]
    ax.plot(lim, lim, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    for mm in ("0.5B", "1.5B", "3B"):
        g = m[m.model == mm]
        if g.empty:
            continue
        ax.scatter(g["eta_hat"], g["eta_test"], s=34, color=MODEL_COLOR[mm],
                   edgecolor="white", linewidth=0.8, zorder=4,
                   label=f"Qwen2.5-{mm}")
    # Label only the points that are separated enough to read; the cluster at
    # the origin is the uninformative-verifier regime and is described in text.
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    for _, r in m.iterrows():
        if max(abs(r["eta_hat"]), abs(r["eta_test"])) < 0.12:
            continue
        ax.annotate(FAM_LABEL.get(r["family"], r["family"]),
                    (r["eta_hat"], r["eta_test"]), textcoords="offset points",
                    xytext=(6, -8), fontsize=6.5, color=MUTED, zorder=6,
                    path_effects=halo)
    ax.annotate("verifier at chance:\n$G=0 \\Rightarrow \\eta=0$", (0.0, 0.0),
                textcoords="offset points", xytext=(10, -34), fontsize=6.5,
                color=MUTED, zorder=6, path_effects=halo,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    from gvgap.metrics import r2 as _r2
    rv = _r2(m["eta_test"].to_numpy(), m["eta_hat"].to_numpy())
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"predicted $\eta$ from $G$ (no fitted parameters)")
    ax.set_ylabel(r"observed $\eta$, held-out (LLM judge)")
    ax.set_title(f"Latent-score model, judge arm  ($R^2={rv:.2f}$)", fontsize=9)
    ax.legend(loc="upper left")
    despine(ax)
    save(fig, "fig8_theory")


if __name__ == "__main__":
    main()
