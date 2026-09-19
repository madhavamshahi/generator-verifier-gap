"""Multi-agent architectures, all built on a shared candidate pool.

Every selector-based architecture (self-consistency, LLM judge, programmatic
verifier, oracle) reads the *same* k candidates for a given task. Differences
between them are therefore attributable purely to selection, with the generator
held fixed -- which is what lets us decompose accuracy into coverage and
selection efficiency.
"""
from __future__ import annotations

import random
import re
from math import gcd
from collections import Counter
from typing import Any, Callable, Sequence

from gvgap import prompts
from gvgap.answers import canonical
from gvgap.engine import Budget, Gen, Runner
from gvgap.verify import check, check_code

BEST = re.compile(r"best\s*[:\-]?\s*(\d+)", re.I)
PAIRPICK = re.compile(r"correct\s*[:\-]?\s*([AB])", re.I)


# ------------------------------------------------------- programmatic checks
def programmatic_verifier(item: dict[str, Any]) -> Callable[[str], bool] | None:
    """A cheap checker derivable from the problem statement alone.

    Critically, none of these consult the gold answer. Where no such checker
    exists for a family we return ``None``; that absence is the phenomenon the
    paper is about, not a gap in the implementation.
    """
    fam = item["family"]
    if fam == "csp":
        mods = item["meta"]["mods"]
        lower = item["meta"]["lower"]

        def _csp(text: str) -> bool:
            from gvgap.verify import _numeric_answer

            v = _numeric_answer(text)
            if v is None:
                return False
            try:
                n = int(float(v))
            except ValueError:
                return False
            if n <= lower or any(n % m != r for m, r in mods):
                return False
            # Minimality is checkable from the statement, but the scan must be
            # bounded: solutions recur with period lcm(m_i), so any candidate
            # beyond lower + period cannot be the smallest, and scanning to an
            # arbitrarily large model-supplied n would not terminate.
            period = 1
            for m, _ in mods:
                period = period * m // gcd(period, m)
            if n > lower + period:
                return False
            for c in range(lower + 1, min(n, lower + period + 1)):
                if all(c % m == r for m, r in mods):
                    return False
            return True

        return _csp
    if fam == "wordcon":
        from gvgap.verify import check_wordcon

        return lambda text: check_wordcon(text, item).correct
    if fam == "mbpp":
        # Only the single test shown in the prompt is available to the
        # verifier; grading uses the full held-out test list. This keeps the
        # verifier realistic and prevents test leakage into the metric.
        visible = {"family": "mbpp", "meta": {"tests": item["meta"]["tests"][:1]}}
        return lambda text: check_code(text, visible).correct
    return None


# ----------------------------------------------------------- candidate pools
def build_pool(
    runner: Runner,
    items: Sequence[dict[str, Any]],
    k: int,
    max_tokens: int,
    temp: float = 0.7,
) -> list[list[Gen]]:
    """k independent samples per item, generated in one flat batch per seed."""
    pools: list[list[Gen]] = [[] for _ in items]
    sysmsg = prompts.SYSTEM[items[0]["family"]]
    for s in range(k):
        gens = runner.chat(
            sysmsg,
            [it["prompt"] for it in items],
            max_tokens=max_tokens,
            temp=temp,
            seeds=[s] * len(items),
        )
        for i, g in enumerate(gens):
            pools[i].append(g)
    return pools


def grade_pool(
    pools: Sequence[Sequence[Gen]], items: Sequence[dict[str, Any]]
) -> list[list[bool]]:
    return [
        [check(g.text, it).correct for g in pool] for pool, it in zip(pools, items)
    ]


# ------------------------------------------------------------------ selectors
def select_first(pool, correct, item, k):
    """Single sample: the k=1 reference point."""
    return 0


def select_oracle(pool, correct, item, k):
    """Upper bound on any selector. Not deployable; used for decomposition."""
    for i in range(k):
        if correct[i]:
            return i
    return 0


def select_majority(pool, correct, item, k):
    """Self-consistency. Returns None when the family has no canonical answer."""
    keys = [canonical(pool[i].text, item) for i in range(k)]
    votes = Counter(x for x in keys if x is not None)
    if not votes:
        return None
    top = votes.most_common(1)[0][0]
    for i in range(k):
        if keys[i] == top:
            return i
    return None


def select_programmatic(pool, correct, item, k):
    ver = programmatic_verifier(item)
    if ver is None:
        return None
    for i in range(k):
        if ver(pool[i].text):
            return i
    return 0  # nothing passes: fall back to the first candidate


# ------------------------------------------------------------------ LLM judge
def judge_select(
    runner: Runner,
    items: Sequence[dict[str, Any]],
    pools: Sequence[Sequence[Gen]],
    k: int,
    max_tokens: int = 160,
    seed: int = 0,
    snippet: int = 700,
) -> tuple[list[int | None], list[Gen], list[list[int]]]:
    """One judge call per task over a randomly permuted candidate list.

    Candidate order is permuted per task with a fixed seed so that judge
    position bias does not align with candidate quality, and the permutation is
    returned so the bias can be measured separately.
    """
    rng = random.Random(seed)
    users, perms = [], []
    for it, pool in zip(items, pools):
        order = list(range(k))
        rng.shuffle(order)
        perms.append(order)
        block = "\n\n".join(
            f"Candidate {j + 1}:\n{pool[order[j]].text.strip()[:snippet]}"
            for j in range(k)
        )
        users.append(prompts.JUDGE_USER.format(problem=it["prompt"], candidates=block))
    gens = runner.chat(
        prompts.JUDGE_SYSTEM, users, max_tokens=max_tokens, temp=0.0,
        seeds=[seed] * len(users),
    )
    picks: list[int | None] = []
    for g, order in zip(gens, perms):
        m = BEST.findall(g.text)
        if not m:
            picks.append(None)
            continue
        j = int(m[-1]) - 1
        picks.append(order[j] if 0 <= j < k else None)
    return picks, gens, perms


# --------------------------------------------- pairwise discrimination probe
def pairwise_gap(
    runner: Runner,
    items: Sequence[dict[str, Any]],
    pools: Sequence[Sequence[Gen]],
    correct: Sequence[Sequence[bool]],
    max_tokens: int = 120,
    seed: int = 0,
    snippet: int = 700,
) -> dict[str, Any]:
    """Measure how often the model prefers a correct candidate over a wrong one.

    This is the cheap diagnostic the paper proposes: it needs only a handful of
    tasks and, unlike measuring lift directly, it never requires building the
    multi-agent system first.

    Each pair is presented **twice, in both orders**, and accuracy is averaged
    over both presentations. Counterbalancing is essential rather than
    cosmetic: a judge that always names the first option scores exactly 0.5
    here, hence G = 0. Randomising the side independently per pair does not
    achieve this -- with a finite sample the correct answer lands on one side
    more often than the other, and a pure position-picker inherits that
    imbalance as spurious discrimination.
    """
    rng = random.Random(seed)
    users, keys, pair_id = [], [], []
    n_pairs = 0
    for idx, (it, pool, cor) in enumerate(zip(items, pools, correct)):
        good = [i for i, c in enumerate(cor) if c]
        bad = [i for i, c in enumerate(cor) if not c]
        if not good or not bad:
            continue  # uninformative: no discriminating pair exists
        gi, bi = rng.choice(good), rng.choice(bad)
        g_txt = pool[gi].text.strip()[:snippet]
        b_txt = pool[bi].text.strip()[:snippet]
        for correct_is_a in (True, False):
            a, b = (g_txt, b_txt) if correct_is_a else (b_txt, g_txt)
            users.append(
                prompts.PAIR_USER.format(problem=it["prompt"], a=a, b=b)
            )
            keys.append("A" if correct_is_a else "B")
            pair_id.append(n_pairs)
        n_pairs += 1
    if not users:
        return {"n_pairs": 0, "n_scored": 0, "acc": None, "G": None,
                "abstain_rate": None, "pick_a_rate": None, "tokens": 0}
    gens = runner.chat(
        prompts.PAIR_SYSTEM, users, max_tokens=max_tokens, temp=0.0,
        seeds=[seed] * len(users),
    )
    hits = abstain = pick_a = scored = 0
    by_pair: dict[int, list[str]] = {}
    for g, key, pid in zip(gens, keys, pair_id):
        m = PAIRPICK.findall(g.text)
        if not m:
            abstain += 1
            continue
        pick = m[-1].upper()
        scored += 1
        pick_a += pick == "A"
        hits += pick == key
        by_pair.setdefault(pid, []).append("correct" if pick == key else "wrong")
    # Order consistency: on how many pairs does the model name the same
    # *candidate* under both orderings? A position-picker scores 0.
    both = [v for v in by_pair.values() if len(v) == 2]
    consistent = sum(1 for v in both if v[0] == v[1])
    acc = hits / scored if scored else None
    return {
        "n_pairs": n_pairs,
        "n_presentations": len(users),
        "n_scored": scored,
        "acc": acc,
        # Gini-style gap: 0 means chance discrimination, 1 means perfect.
        "G": None if acc is None else 2 * acc - 1,
        "abstain_rate": abstain / len(users),
        "pick_a_rate": pick_a / scored if scored else None,
        "order_consistency": consistent / len(both) if both else None,
        "tokens": sum(g.total_tokens for g in gens),
    }


# ------------------------------------------------- sequential (single-agent)
def seq_refine(
    runner: Runner,
    items: Sequence[dict[str, Any]],
    rounds: int,
    max_tokens: int,
    temp: float = 0.7,
    seed: int = 0,
) -> tuple[list[Gen], list[Budget]]:
    """Token-matched single-agent baseline: one agent revising its own answer.

    This is the comparison most multi-agent papers omit. Spending the same
    tokens serially inside one context is the honest alternative to spending
    them across parallel agents.
    """
    sysmsg = prompts.SYSTEM[items[0]["family"]]
    budgets = [Budget() for _ in items]
    cur = runner.chat(
        sysmsg, [it["prompt"] for it in items], max_tokens=max_tokens,
        temp=temp, seeds=[seed] * len(items),
    )
    for b, g in zip(budgets, cur):
        b.add(g)
    for r in range(1, rounds):
        users = [
            prompts.REFINE_USER.format(problem=it["prompt"], previous=g.text.strip()[:1200])
            for it, g in zip(items, cur)
        ]
        cur = runner.chat(
            sysmsg, users, max_tokens=max_tokens, temp=temp,
            seeds=[seed * 100 + r] * len(items),
        )
        for b, g in zip(budgets, cur):
            b.add(g)
    return list(cur), budgets


# --------------------------------------------------------------- debate
def debate(
    runner: Runner,
    items: Sequence[dict[str, Any]],
    n_agents: int,
    rounds: int,
    max_tokens: int,
    temp: float = 0.7,
    seed: int = 0,
    snippet: int = 500,
) -> tuple[list[list[Gen]], list[Budget]]:
    """Multi-agent debate: agents see each other's answers and may revise."""
    sysmsg = prompts.SYSTEM[items[0]["family"]]
    budgets = [Budget() for _ in items]
    # Round 0: independent answers.
    cur: list[list[Gen]] = [[] for _ in items]
    for a in range(n_agents):
        gens = runner.chat(
            sysmsg, [it["prompt"] for it in items], max_tokens=max_tokens,
            temp=temp, seeds=[seed * 10 + a] * len(items),
        )
        for i, g in enumerate(gens):
            cur[i].append(g)
            budgets[i].add(g)
    for r in range(1, rounds):
        nxt: list[list[Gen]] = [[] for _ in items]
        for a in range(n_agents):
            users = []
            for it, pool in zip(items, cur):
                others = "\n\n".join(
                    f"Solver {j + 1}: {pool[j].text.strip()[:snippet]}"
                    for j in range(n_agents) if j != a
                )
                users.append(prompts.DEBATE_USER.format(problem=it["prompt"], others=others))
            gens = runner.chat(
                sysmsg, users, max_tokens=max_tokens, temp=temp,
                seeds=[seed * 1000 + r * 10 + a] * len(items),
            )
            for i, g in enumerate(gens):
                nxt[i].append(g)
                budgets[i].add(g)
        cur = nxt
    return cur, budgets


# ------------------------------------------- verifier-free aggregation probe
def plurality_alignment(
    items: Sequence[dict[str, Any]],
    pools: Sequence[Sequence[Gen]],
    correct: Sequence[Sequence[bool]],
    k: int,
) -> dict[str, Any]:
    """How well the *modal* answer tracks the truth, independent of any verifier.

    Self-consistency needs no verifier at all, so its gains cannot be explained
    by the generator-verifier gap. It works when the sampling distribution is
    peaked on the correct answer. We measure that directly: the share of mass
    on the modal answer, and how often that mode is correct. Together with G
    these are the two cheap diagnostics the paper proposes.
    """
    shares, mode_correct, applicable = [], [], 0
    for it, pool, cor in zip(items, pools, correct):
        keys = [canonical(pool[i].text, it) for i in range(k)]
        votes = Counter(x for x in keys if x is not None)
        if not votes:
            continue
        applicable += 1
        top, n_top = votes.most_common(1)[0]
        shares.append(n_top / k)
        idx = next(i for i in range(k) if keys[i] == top)
        mode_correct.append(bool(cor[idx]))
    if not applicable:
        return {"applicable": 0, "modal_share": None, "plurality_alignment": None}
    p1 = sum(c[0] for c in correct) / len(correct)
    align = sum(mode_correct) / len(mode_correct)
    return {
        "applicable": applicable,
        "modal_share": sum(shares) / len(shares),
        "mode_accuracy": align,
        # Signed advantage of the mode over a single random sample: the
        # verifier-free analogue of G.
        "plurality_alignment": align - p1,
    }


def seq_long(
    runner: Runner,
    items: Sequence[dict[str, Any]],
    budget_mult: int,
    max_tokens: int,
    temp: float = 0.7,
    seed: int = 0,
) -> tuple[list[Gen], list[Budget]]:
    """Single agent, one call, asked to reason at length with a larger cap.

    Self-refinement is not the only way to spend serial compute, and models are
    known to be poor at correcting their own reasoning. Reporting a long
    chain-of-thought arm as well means the single-agent frontier is the upper
    envelope of two methods rather than the fortunes of one.
    """
    sysmsg = prompts.SYSTEM[items[0]["family"]]
    users = [it["prompt"] + prompts.LONG_SUFFIX for it in items]
    gens = runner.chat(
        sysmsg, users, max_tokens=max_tokens * budget_mult, temp=temp,
        seeds=[seed] * len(items),
    )
    budgets = []
    for g in gens:
        b = Budget()
        b.add(g)
        budgets.append(b)
    return list(gens), budgets
