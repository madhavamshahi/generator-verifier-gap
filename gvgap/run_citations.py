"""Case study: reference generation as a selection problem.

The task is "produce a real published reference on topic X". A generator
proposes k candidates; a selector picks one. Two selectors are compared:

  * the same language model asked whether a reference is real (what a
    critic-agent design would use), and
  * a lookup against the OpenAlex registry (an external oracle).

Both are scored against the same ground truth, so eta is directly comparable to
the six task families in the main experiment.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap import citations as CIT
from gvgap.engine import Runner
from gvgap.metrics import boot, pass_at_k
from gvgap.theory import predict_eta

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
K = 8
N_REFS = 3          # references requested per generation call
MATCH_THRESHOLD = 0.88


def is_real(ref: CIT.Ref, oa: CIT.OpenAlex) -> tuple[bool | None, float | None]:
    """Ground truth: does a work with essentially this title exist?

    Returns ``None`` when the registry could not be reached, so that a network
    failure is never recorded as a fabricated citation.
    """
    m = oa.best_match(ref.title)
    if m.get("unresolved") or m.get("ratio") is None:
        return None, None
    return m["ratio"] >= MATCH_THRESHOLD, m["ratio"]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    r = Runner(model, batch_size=32)
    oa = CIT.Registry()

    # ---- 1. generate candidate references -----------------------------
    topics = CIT.TOPICS
    users = [CIT.GEN_USER.format(n=N_REFS, topic=t) for t in topics]
    pools: list[list[CIT.Ref]] = [[] for _ in topics]
    for s in range(K):
        gens = r.chat(CIT.GEN_SYSTEM, users, max_tokens=256, temp=0.8,
                      seeds=[s] * len(users))
        for i, g in enumerate(gens):
            refs = CIT.parse_refs(g.text, topics[i])
            # One candidate per draw keeps the pool structure identical to the
            # main experiment: k independent attempts at the same task.
            pools[i].append(refs[0] if refs else CIT.Ref(topics[i], "", "", "", ""))
    print(f"generated {sum(len(p) for p in pools)} candidate references",
          flush=True)

    # ---- 2. ground truth via OpenAlex ---------------------------------
    truth: list[list[bool | None]] = []
    ratios: list[list[float | None]] = []
    for i, pool in enumerate(pools):
        row, rr = [], []
        for ref in pool:
            if not ref.title:
                # An empty generation is a genuine failure to produce a
                # reference, which is different from an unresolvable one.
                row.append(False); rr.append(0.0); continue
            ok, ratio = is_real(ref, oa)
            row.append(ok); rr.append(ratio)
        truth.append(row); ratios.append(rr)
        if (i + 1) % 10 == 0:
            oa.save()
            print(f"  resolved {i+1}/{len(pools)} topics "
                  f"({oa.calls} API calls)", flush=True)
    oa.save()

    flat_all = [t for row in truth for t in row]
    n_unresolved = sum(1 for t in flat_all if t is None)
    flat = [t for t in flat_all if t is not None]
    print(f"resolved {len(flat)}/{len(flat_all)} candidates "
          f"({n_unresolved} unresolved, excluded); "
          f"throttled {oa.throttled}, failed {oa.failed}", flush=True)
    print(f"hallucination rate: {1 - np.mean(flat):.3f}", flush=True)
    # Topics are usable only if every candidate resolved, so that p1, coverage
    # and the selectors are all computed over the same complete pools.
    usable = [i for i in range(len(topics))
              if all(t is not None for t in truth[i])]
    print(f"usable topics: {len(usable)}/{len(topics)}", flush=True)
    if len(usable) < 10:
        print("too few fully-resolved topics; aborting case study", flush=True)
        oa.save()
        return

    # ---- 3. model verifier: binary verdict per reference ---------------
    vq, vidx = [], []
    for i in usable:
        pool = pools[i]
        for j, ref in enumerate(pool):
            if not ref.title:
                continue
            vq.append(CIT.VERIFY_USER.format(
                title=ref.title, author=ref.author or "unknown",
                year=ref.year or "unknown"))
            vidx.append((i, j))
    vg = r.chat(CIT.VERIFY_SYSTEM, vq, max_tokens=120, temp=0.0,
                seeds=[0] * len(vq))
    verdict: dict[tuple[int, int], bool | None] = {}
    vtokens = 0
    for (i, j), g in zip(vidx, vg):
        m = CIT.VERDICT.findall(g.text)
        verdict[(i, j)] = (m[-1].upper() == "REAL") if m else None
        vtokens += g.total_tokens

    yes = [(verdict[(i, j)], truth[i][j]) for (i, j) in vidx
           if verdict.get((i, j)) is not None]
    tp = sum(1 for p, t in yes if p and t)
    fp = sum(1 for p, t in yes if p and not t)
    tn = sum(1 for p, t in yes if not p and not t)
    fn = sum(1 for p, t in yes if not p and t)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    bal = (tpr + tnr) / 2
    say_real = sum(1 for p, _ in yes if p) / len(yes) if yes else float("nan")
    print(f"model verifier: TPR={tpr:.3f} TNR={tnr:.3f} balanced={bal:.3f} "
          f"says-real={say_real:.3f}", flush=True)

    # ---- 4. selection under each verifier ------------------------------
    def select_llm(i: int) -> int:
        for j in range(K):
            if verdict.get((i, j)) is True:
                return j
        return 0

    def select_oracle_api(i: int) -> int:
        for j in range(K):
            if truth[i][j]:
                return j
        return 0

    p1 = float(np.mean([truth[i][0] for i in usable]))
    cov = float(np.mean([
        pass_at_k(K, int(sum(truth[i])), K) for i in usable
    ]))
    acc_llm = float(np.mean([truth[i][select_llm(i)] for i in usable]))
    acc_api = float(np.mean([truth[i][select_oracle_api(i)] for i in usable]))
    denom = cov - p1
    eta_llm = (acc_llm - p1) / denom if abs(denom) > 1e-9 else float("nan")
    eta_api = (acc_api - p1) / denom if abs(denom) > 1e-9 else float("nan")

    # ---- 5. pairwise gap, same probe as the main experiment ------------
    rng = np.random.default_rng(5)
    pq, pkey = [], []
    for i in usable:
        good = [j for j in range(K) if truth[i][j] and pools[i][j].title]
        bad = [j for j in range(K) if not truth[i][j] and pools[i][j].title]
        if not good or not bad:
            continue
        gj, bj = int(rng.choice(good)), int(rng.choice(bad))
        a_is_correct = bool(rng.random() < 0.5)
        A, B = ((pools[i][gj], pools[i][bj]) if a_is_correct
                else (pools[i][bj], pools[i][gj]))
        fmt = lambda x: f"Title: {x.title}\nFirst author: {x.author}\nYear: {x.year}"
        pq.append(CIT.PAIR_USER.format(
            problem=f"Which of these is a real published paper on "
                    f"{topics[i]}?",
            a=fmt(A), b=fmt(B)))
        pkey.append("A" if a_is_correct else "B")
    G = float("nan")
    if pq:
        pg = r.chat(CIT.PAIR_SYSTEM, pq, max_tokens=100, temp=0.0,
                    seeds=[0] * len(pq))
        hits = scored = 0
        for g, key in zip(pg, pkey):
            m = re.findall(r"correct\s*[:\-]?\s*([AB])", g.text, re.I)
            if not m:
                continue
            scored += 1
            hits += m[-1].upper() == key
        G = 2 * (hits / scored) - 1 if scored else float("nan")

    counts = [int(sum(truth[i])) for i in usable]
    pred_acc, pred_eta = (float("nan"), float("nan"))
    if not np.isnan(G):
        pred_acc, pred_eta = predict_eta(counts, K, G, p1, cov)

    out = {
        "model": model, "n_topics": len(usable),
        "n_topics_attempted": len(topics),
        "n_unresolved_candidates": n_unresolved,
        "openalex_throttled": oa.throttled, "openalex_failed": oa.failed,
        "k": K,
        "match_threshold": MATCH_THRESHOLD,
        "hallucination_rate": float(1 - np.mean(flat)),
        "p1": p1, "coverage": cov, "headroom": cov - p1,
        "acc_llm_verifier": acc_llm, "acc_api_verifier": acc_api,
        "eta_llm_verifier": eta_llm, "eta_api_verifier": eta_api,
        "G_pairwise": G, "predicted_eta_from_G": pred_eta,
        "predicted_acc_from_G": pred_acc,
        "verifier_tpr": tpr, "verifier_tnr": tnr,
        "verifier_balanced_acc": bal, "verifier_says_real_rate": say_real,
        "llm_verifier_tokens": vtokens,
        "api_verifier_tokens": 0,
        "registry_calls": oa.calls, "registry_by_source": oa.by_source,
    }
    (RES / "citation_case_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
