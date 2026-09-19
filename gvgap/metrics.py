"""Estimators and uncertainty for the paper's central quantities.

Three things matter for the claims we make:

* ``pass_at_k`` uses the unbiased combinatorial estimator rather than the naive
  "any of k correct", so coverage curves from a single pool of n samples are
  comparable across k.
* every reported difference carries a cluster bootstrap CI over *tasks*, since
  samples within a task are not independent.
* architecture comparisons are paired on the same task set, so we use a paired
  bootstrap rather than treating the two arms as independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Sequence

import numpy as np

RNG = np.random.default_rng(20260919)
N_BOOT = 10_000


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one correct among k draws) given c/n correct."""
    if k > n:
        raise ValueError(f"k={k} exceeds pool size n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def coverage_curve(correct: Sequence[Sequence[bool]], kmax: int) -> np.ndarray:
    """Mean pass@k over tasks, for k = 1..kmax."""
    out = np.zeros(kmax)
    for k in range(1, kmax + 1):
        out[k - 1] = float(
            np.mean([pass_at_k(len(c), int(sum(c)), k) for c in correct])
        )
    return out


@dataclass
class Est:
    point: float
    lo: float
    hi: float
    n: int

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{self.point:.3f} [{self.lo:.3f},{self.hi:.3f}]"


def boot(
    values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = N_BOOT, alpha: float = 0.05, rng: np.random.Generator = RNG,
) -> Est:
    """Percentile bootstrap over tasks."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return Est(float("nan"), float("nan"), float("nan"), 0)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    draws = np.array([stat(a[i]) for i in idx])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Est(float(stat(a)), float(lo), float(hi), a.size)


def paired_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT,
    alpha: float = 0.05, rng: np.random.Generator = RNG,
) -> tuple[Est, float]:
    """Paired bootstrap of mean(a) - mean(b), plus a two-sided bootstrap p.

    The p-value is the usual bootstrap tail probability that the paired
    difference has the opposite sign, doubled.
    """
    x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if x.size == 0:
        return Est(float("nan"), float("nan"), float("nan"), 0), float("nan")
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    draws = x[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    centred = draws - draws.mean()
    p = 2.0 * min(
        float(np.mean(centred >= abs(x.mean()))),
        float(np.mean(centred <= -abs(x.mean()))),
    )
    return Est(float(x.mean()), float(lo), float(hi), x.size), min(1.0, p)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def selection_efficiency(
    acc_sel: float, p1: float, cov: float
) -> float:
    """eta = fraction of the available coverage headroom a selector converts.

    eta = 0 means the selector does no better than picking at random among the
    candidates; eta = 1 means it matches an oracle. Undefined when there is no
    headroom (cov == p1), which we surface as NaN rather than silently zero.
    """
    denom = cov - p1
    if abs(denom) < 1e-9:
        return float("nan")
    return (acc_sel - p1) / denom


def interp_accuracy(tokens: Sequence[float], acc: Sequence[float], at: float) -> float:
    """Accuracy of a baseline curve at a given token budget.

    Used to price the single-agent baseline at exactly the multi-agent system's
    cost. Outside the measured range we clamp to the nearest endpoint rather
    than extrapolating, which is the conservative choice: it never invents
    baseline strength we did not observe.
    """
    t = np.asarray(tokens, dtype=float)
    a = np.asarray(acc, dtype=float)
    o = np.argsort(t)
    t, a = t[o], a[o]
    if at <= t[0]:
        return float(a[0])
    if at >= t[-1]:
        return float(a[-1])
    return float(np.interp(at, t, a))


def r2(y: Sequence[float], yhat: Sequence[float]) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
