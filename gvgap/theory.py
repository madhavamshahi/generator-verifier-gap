"""A latent-score model linking pairwise discrimination to listwise selection.

The cheap diagnostic G is *pairwise*: how often the verifier prefers a correct
candidate to an incorrect one. What an ensemble actually does is *listwise*:
pick one of k. This module supplies the link, so that G yields a parameter-free
prediction of selection efficiency rather than a quantity merely correlated
with it.

Model. The verifier assigns each candidate a latent score, drawn from
N(delta, 1) for a correct candidate and N(0, 1) for an incorrect one, and
selects the argmax. Then the pairwise preference probability is

    q = P(N(delta,1) > N(0,1)) = Phi(delta / sqrt(2)),

so an observed q identifies delta = sqrt(2) * Phi^{-1}(q), with no free
parameters left. Given delta and the observed distribution of how many of the
k candidates are correct, the listwise selection probability follows by
integration, and eta follows from the decomposition.

The model is deliberately the simplest one that respects the pairwise/listwise
distinction; Section 6 reports where it holds and where it breaks.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(31337)


def delta_from_G(G: float) -> float:
    """Latent-score separation implied by a pairwise discrimination G."""
    q = float(np.clip((1.0 + G) / 2.0, 1e-6, 1 - 1e-6))
    return float(np.sqrt(2.0) * norm.ppf(q))


def p_select_correct(c: int, k: int, delta: float, n_mc: int = 40_000,
                     rng: np.random.Generator | None = None) -> float:
    """P(argmax score is a correct candidate) with c of k correct."""
    if c <= 0:
        return 0.0
    if c >= k:
        return 1.0
    rng = rng or RNG
    good = rng.normal(delta, 1.0, size=(n_mc, c)).max(axis=1)
    bad = rng.normal(0.0, 1.0, size=(n_mc, k - c)).max(axis=1)
    return float(np.mean(good > bad))


def predict_accuracy(
    correct_counts: list[int], k: int, G: float, n_mc: int = 40_000
) -> float:
    """Expected selector accuracy over the observed pool-composition mixture.

    ``correct_counts`` is the per-task number of correct candidates in the pool,
    which is observable on the calibration split without building the
    multi-agent system.
    """
    delta = delta_from_G(G)
    cache: dict[int, float] = {}
    tot = 0.0
    for c in correct_counts:
        c = int(min(max(c, 0), k))
        if c not in cache:
            cache[c] = p_select_correct(c, k, delta, n_mc)
        tot += cache[c]
    return tot / len(correct_counts)


def predict_eta(
    correct_counts: list[int], k: int, G: float, p1: float, coverage: float,
    n_mc: int = 40_000,
) -> tuple[float, float]:
    """(predicted selector accuracy, predicted eta) from G alone."""
    acc = predict_accuracy(correct_counts, k, G, n_mc)
    denom = coverage - p1
    eta = (acc - p1) / denom if abs(denom) > 1e-9 else float("nan")
    return acc, eta


if __name__ == "__main__":
    # Sanity: a chance verifier recovers eta = 0; a perfect one recovers 1.
    k = 8
    counts = [0, 1, 2, 3, 4, 5, 6, 7, 8] * 20
    p1 = float(np.mean([c / k for c in counts]))
    cov = float(np.mean([1.0 if c > 0 else 0.0 for c in counts]))
    for G in (0.0, 0.3, 0.6, 0.9, 0.99):
        acc, eta = predict_eta(counts, k, G, p1, cov, n_mc=20000)
        print(f"G={G:4.2f}  delta={delta_from_G(G):5.2f}  acc={acc:.3f}  eta={eta:.3f}")
