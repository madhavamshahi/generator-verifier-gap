"""Programmatic generators for the three synthetic task families.

All generators are seeded and deterministic, so the released task files can be
regenerated bit-for-bit. Synthetic families also carry zero benchmark-
contamination risk, which matters because we compare against pretrained models
whose training corpora we cannot inspect.
"""
from __future__ import annotations

import random
from math import gcd
from typing import Any

# ---------------------------------------------------------------- CSP family


def _crt_smallest(mods: list[tuple[int, int]], lower: int) -> int | None:
    """Smallest n > lower satisfying n % m == r for every (m, r).

    Brute force over the lcm period. Periods are kept small by construction so
    this stays exact rather than relying on a CRT implementation we would also
    have to trust.
    """
    period = 1
    for m, _ in mods:
        period = period * m // gcd(period, m)
    start = lower + 1
    for n in range(start, start + period + 1):
        if all(n % m == r for m, r in mods):
            return n
    return None


def gen_csp(n_items: int, seed: int = 0) -> list[dict[str, Any]]:
    """Congruence search: find the smallest n above a bound hitting k residues.

    Verification is a handful of modulo operations; generation requires either
    the CRT or a search. This is the extreme high-asymmetry anchor of our task
    suite.
    """
    rng = random.Random(seed)
    primes = [3, 4, 5, 6, 7, 8, 9]
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    while len(out) < n_items:
        k = 2
        mods = rng.sample(primes, k)
        pairs = [(m, rng.randrange(m)) for m in mods]
        lower = rng.choice([10, 20, 30, 50, 100])
        key = (tuple(sorted(pairs)), lower)
        if key in seen:
            continue
        ans = _crt_smallest(pairs, lower)
        if ans is None or ans > lower + 5000:
            continue
        seen.add(key)
        conds = " and ".join(
            f"n leaves remainder {r} when divided by {m}" for m, r in pairs
        )
        out.append(
            {
                "family": "csp",
                "prompt": (
                    f"Find the smallest integer n greater than {lower} such that "
                    f"{conds}."
                ),
                "answer": str(ans),
                "meta": {"mods": pairs, "lower": lower, "k": k},
            }
        )
    return out


# -------------------------------------------------------------- logic family

_REL = {
    "height": ("taller than", "shorter than"),
    "age": ("older than", "younger than"),
    "speed": ("faster than", "slower than"),
    "weight": ("heavier than", "lighter than"),
}
_NAMES = [
    "Ana", "Ben", "Cara", "Dev", "Elif", "Femi", "Gus", "Hana",
    "Ivo", "Jun", "Kai", "Lena",
]


def gen_logic(n_items: int, seed: int = 0) -> list[dict[str, Any]]:
    """Transitive ordering puzzles over a shuffled chain of comparisons.

    A verifier cannot shortcut this: confirming a proposed extremum means
    re-running essentially the same chain of deductions the generator ran, so
    we expect verification to be about as expensive as generation. This is the
    low-asymmetry anchor.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for _ in range(n_items):
        n = rng.choice([4, 5, 5, 6])
        names = rng.sample(_NAMES, n)
        dim = rng.choice(list(_REL))
        more, less = _REL[dim]
        # names[0] is the true maximum, names[-1] the true minimum.
        facts = []
        for i in range(n - 1):
            if rng.random() < 0.5:
                facts.append(f"{names[i]} is {more} {names[i + 1]}.")
            else:
                facts.append(f"{names[i + 1]} is {less} {names[i]}.")
        rng.shuffle(facts)
        ask_max = rng.random() < 0.5
        target, question = (
            (names[0], f"Who is the {more.split()[0]}?")
            if ask_max
            else (names[-1], f"Who is the {less.split()[0]}?")
        )
        out.append(
            {
                "family": "logic",
                "prompt": " ".join(facts) + " " + question,
                "answer": target,
                "meta": {"n_entities": n, "dim": dim, "candidates": sorted(names)},
            }
        )
    return out


# ------------------------------------------------------------ wordcon family

_TOPICS = [
    ("garden", ["soil", "bloom"]), ("harbour", ["vessel", "tide"]),
    ("library", ["volume", "silence"]), ("kitchen", ["simmer", "spice"]),
    ("mountain", ["ridge", "summit"]),
    ("market", ["barter", "crowd"]), ("desert", ["dune", "mirage"]),
    ("factory", ["piston", "shift"]), ("river", ["current", "bank"]),
    ("forest", ["canopy", "moss"]), ("studio", ["canvas", "pigment"]),
    ("clinic", ["patient", "chart"]),
]


def gen_wordcon(n_items: int, seed: int = 0) -> list[dict[str, Any]]:
    """Constrained sentence writing: exact length plus required lexical items.

    Checking is a token count and two substring tests. Producing a sentence
    that satisfies both constraints simultaneously is much harder, especially
    for small models that cannot reliably count their own tokens. This gives us
    a high-asymmetry task in a natural-language rather than symbolic domain.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for _ in range(n_items):
        topic, words = rng.choice(_TOPICS)
        length = rng.choice([6, 7, 8, 9])
        out.append(
            {
                "family": "wordcon",
                "prompt": (
                    f"Write one sentence about a {topic}. It must be exactly "
                    f"{length} words long and must contain both of these words: "
                    f"'{words[0]}' and '{words[1]}'."
                ),
                "answer": "",  # checked structurally, not by string match
                "meta": {"length": length, "required": words, "topic": topic},
            }
        )
    return out
