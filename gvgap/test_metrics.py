"""Correctness tests for the estimators the paper's claims rest on.

These are checked against brute-force or analytic ground truth rather than
against themselves, since a subtly wrong estimator would propagate silently
into every reported number.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap.metrics import boot, holm, paired_diff, pass_at_k, selection_efficiency
from gvgap.theory import delta_from_G, p_select_correct


def test_pass_at_k_bruteforce() -> None:
    """pass@k must equal the exact probability over all k-subsets."""
    for n in range(1, 9):
        for c in range(0, n + 1):
            labels = [True] * c + [False] * (n - c)
            for k in range(1, n + 1):
                subs = list(itertools.combinations(range(n), k))
                exact = sum(any(labels[i] for i in s) for s in subs) / len(subs)
                got = pass_at_k(n, c, k)
                assert abs(exact - got) < 1e-12, (n, c, k, exact, got)
    print("pass_at_k matches exhaustive enumeration for all n<=8  OK")


def test_pass_at_k_edges() -> None:
    assert pass_at_k(8, 0, 4) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert abs(pass_at_k(8, 1, 1) - 0.125) < 1e-12
    try:
        pass_at_k(4, 2, 5)
    except ValueError:
        print("pass_at_k rejects k>n  OK")
    else:
        raise AssertionError("expected ValueError for k>n")


def test_bootstrap_coverage() -> None:
    """A 95% interval should cover the truth close to 95% of the time."""
    rng = np.random.default_rng(0)
    p, n, trials = 0.4, 120, 400
    hits = 0
    for _ in range(trials):
        x = rng.random(n) < p
        e = boot(x.astype(float), n_boot=600, rng=rng)
        hits += e.lo <= p <= e.hi
    cov = hits / trials
    assert 0.90 <= cov <= 0.99, cov
    print(f"bootstrap CI empirical coverage {cov:.3f} (nominal 0.95)  OK")


def test_paired_diff_null() -> None:
    """Under the null the p-value should be roughly uniform, not anti-conservative."""
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(300):
        a = rng.random(80)
        b = a + rng.normal(0, 0.1, 80)   # zero true difference
        ps.append(paired_diff(a, b, n_boot=400, rng=rng)[1])
    rate = float(np.mean(np.array(ps) < 0.05))
    assert rate < 0.12, rate
    print(f"paired bootstrap false-positive rate at alpha=.05: {rate:.3f}  OK")


def test_holm() -> None:
    # Sorted p = (0.01, 0.03, 0.04) with m=3: multipliers 3, 2, 1 and a running
    # maximum give adjusted (0.03, 0.06, 0.06), mapped back to input order.
    assert [round(x, 6) for x in holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]
    assert all(a >= b for a, b in zip(holm([0.2, 0.2, 0.2]), [0.2, 0.2, 0.2]))
    assert max(holm([0.5, 0.6])) <= 1.0
    print("Holm-Bonferroni matches hand-computed values and is capped at 1  OK")


def test_eta_bounds() -> None:
    assert abs(selection_efficiency(0.5, 0.5, 0.9) - 0.0) < 1e-12
    assert abs(selection_efficiency(0.9, 0.5, 0.9) - 1.0) < 1e-12
    assert selection_efficiency(0.3, 0.5, 0.9) < 0        # anti-correlated
    assert np.isnan(selection_efficiency(0.5, 0.5, 0.5))  # no headroom
    print("selection efficiency hits 0 at chance, 1 at oracle, <0 when harmful  OK")


def test_theory_limits() -> None:
    """The latent-score model must respect its own boundary conditions."""
    assert p_select_correct(0, 8, 1.0) == 0.0
    assert p_select_correct(8, 8, 1.0) == 1.0
    # With no discrimination, selection is uniform: c of k correct -> c/k.
    for c in (1, 3, 5):
        got = p_select_correct(c, 8, delta_from_G(0.0), n_mc=60000)
        assert abs(got - c / 8) < 0.02, (c, got)
    # Monotone in delta.
    prev = -1.0
    for G in (-0.5, 0.0, 0.3, 0.6, 0.9):
        v = p_select_correct(3, 8, delta_from_G(G), n_mc=40000)
        assert v > prev - 0.01, (G, v, prev)
        prev = v
    print("latent-score model: correct limits, uniform at G=0, monotone in G  OK")


if __name__ == "__main__":
    test_pass_at_k_bruteforce()
    test_pass_at_k_edges()
    test_bootstrap_coverage()
    test_paired_diff_null()
    test_holm()
    test_eta_bounds()
    test_theory_limits()
    print("\nall metric tests passed")
