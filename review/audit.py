"""Reproduce every number in review/REVIEW.md from the repo's shipped results.

Run from the repo root:  python3 review/audit.py
Needs numpy, pandas, scipy (already in requirements.txt). No GPU, no network.
Finding IDs (V1, S2, ...) match review/REVIEW.md.
"""
import collections
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gvgap.analysis import upper_envelope  # noqa: E402
from gvgap.metrics import interp_accuracy  # noqa: E402

RES = REPO / "results"
RAW = sorted(glob.glob(str(RES / "raw" / "*" / "*.json")))


def load(p):
    r = json.load(open(p))
    return r, Path(p).parent.name.split("-")[1]


def prompts(fam):
    return {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(REPO / "data" / f"{fam}.jsonl")}


def r2(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def head(t):
    print(f"\n== {t}")


def v1_truncation():
    head("V1  pool samples hitting the max-token cap")
    for p in RAW:
        r, m = load(p)
        ct = np.array([t["pool_completion_tokens"] for t in r["tasks"]])
        ok = np.array([t["pool_correct"] for t in r["tasks"]])
        tr = ct >= r["max_tokens"]
        if tr.mean() >= 0.05:
            print(f"{m} {r['family']:6s} cap={r['max_tokens']} truncated={tr.mean():.0%} "
                  f"acc(truncated)={ok[tr].mean():.3f} acc(complete)={ok[~tr].mean():.3f}")


def matched(arms):
    s = pd.read_csv(RES / "summary.csv")
    rows = []
    for (m, f), g in s.groupby(["model", "family"]):
        one = g[g.arch == "single"].iloc[0]
        fr = g[g.arch.isin(arms)]
        st, sa = upper_envelope(np.r_[one.tokens, fr.tokens], np.r_[one.acc, fr.acc])
        for r in g.itertuples():
            if r.arch not in ("seq", "long", "single", "oracle"):
                rows.append(dict(family=f, arch=r.arch, naive=r.acc - one.acc,
                                 matched=r.acc - interp_accuracy(st, sa, r.tokens)))
    return pd.DataFrame(rows)


def v1_v2_headline():
    head("V1/V2  headline token-matched stats under defensible alternatives")

    def show(d, label):
        pos = d.naive > 0
        flip = pos & (d.matched <= 0)
        print(f"{label:44s} flipped={flip.sum()}/{pos.sum()} ({flip.sum() / pos.sum():.0%}) "
              f"median naive={d.naive.median():+.3f} matched={d.matched.median():+.3f} "
              f"net-negative={(d.matched < 0).mean():.0%}")

    both, seq = matched(["seq", "long"]), matched(["seq"])
    show(both, "paper (refine + long-CoT frontier)")
    show(both[both.family != "csp"], "paper frontier, CSP excluded")
    show(seq, "Sec.3 definition (refinement only)")
    show(seq[seq.family != "csp"], "refinement only, CSP excluded")
    ci = pd.read_csv(RES / "matched_lift_ci.csv")
    neg = ci[(ci.p_holm < 0.05) & (ci.matched_lift < 0)]
    print(f"Holm-significant negative cells: {len(neg)}, of which CSP: {(neg.family == 'csp').sum()}")


def v3_winners_curse():
    head("V3  optimism of an in-sample max frontier (pick arm on calib, score on test)")
    arms = ["seq_r1", "seq_r2", "seq_r3", "seq_r4", "long_m2", "long_m4"]
    gaps = []
    for p in RAW:
        r, _ = load(p)

        def acc(split):
            T = [t for t in r["tasks"] if t["split"] == split]
            out = {"single": np.mean([t["pool_correct"][0] for t in T])}
            out.update({a: np.mean([t["arch"][a]["acc"] for t in T]) for a in arms})
            return out

        cal, tst = acc("calib"), acc("test")
        gaps.append(max(tst.values()) - tst[max(cal, key=cal.get)])
    print(f"mean optimism {np.mean(gaps):+.3f}, max {np.max(gaps):+.3f}")


def v4_clamping():
    head("V4  multi-agent cells priced beyond the largest measured single-agent budget")
    s = pd.read_csv(RES / "summary.csv")
    ml = pd.read_csv(RES / "matched_lift.csv")
    fmax = s[s.arch.isin(["single", "seq", "long"])].groupby(["model", "family"]).tokens.max()
    ratio = ml.apply(lambda r: r.tokens / fmax[(r.model, r.family)], axis=1)
    b = ratio > 1
    print(f"{b.sum()}/{len(ml)} cells; ratio median {ratio[b].median():.2f}, max {ratio[b].max():.2f}")


def v6_snippet():
    head("V6  G-probe pairs (paper's Random(11)) with a candidate longer than the 700-char view")
    for p in RAW:
        r, m = load(p)
        if r["family"] not in ("gsm8k", "csp"):
            continue
        rng, n, a, b = random.Random(11), 0, 0, 0
        for t in (t for t in r["tasks"] if t["split"] == "calib"):
            good = [i for i, c in enumerate(t["pool_correct"]) if c]
            bad = [i for i, c in enumerate(t["pool_correct"]) if not c]
            if good and bad:
                gi, bi = rng.choice(good), rng.choice(bad)
                L = max(t["pool_completion_tokens"][gi], t["pool_completion_tokens"][bi])
                n, a, b = n + 1, a + (L > 175), b + (L > 230)
        print(f"{m} {r['family']:6s} pairs={n:2d}  >175 tok: {a}  >230 tok: {b}")


def s1_prediction():
    head("S1  what drives R^2 = 0.84")
    p = pd.read_csv(RES / "prediction.csv")
    print(f"paper R^2 = {r2(p.actual, p.pred_apriori):.3f} over {len(p)} cells")
    for a, g in p.groupby("arch"):
        print(f"  {a:5s} n={len(g):2d} R^2={r2(g.actual, g.pred_apriori):+.3f}")
    print(f"ignore G (judge predicted 0): R^2 = {r2(p.actual, p.pred_apriori.where(p.arch != 'judge', 0.0)):.3f}")


def s2_s3_gap():
    head("S2/S3  calibration G with 95% Clopper-Pearson CI over pairs")
    for p in RAW:
        r, m = load(p)
        d = r["diagnostics"]["calib"]["gap"]
        n, x = d["n_pairs"], round(d["acc"] * d["n_pairs"])
        lo = beta.ppf(.025, x, n - x + 1) if x else 0.0
        hi = beta.ppf(.975, x + 1, n - x) if x < n else 1.0
        print(f"{m} {r['family']:7s} pairs={n:2d} G={d['G']:+.2f} CI=[{2 * lo - 1:+.2f},{2 * hi - 1:+.2f}] "
              f"pickA={d['pick_a_rate']:.2f} order-consistency={d['order_consistency']:.2f} "
              f"abstain={d['abstain_rate']:.2f}")


def s5_r3_duplicates():
    head("S5/R3  WordCon duplicate prompts: leakage, identical pools, seq_r1 != single")
    pr = prompts("wordcon")
    for p in RAW:
        r, m = load(p)
        if r["family"] != "wordcon":
            continue
        cnt = collections.Counter(pr[t["id"]] for t in r["tasks"])
        groups = collections.defaultdict(list)
        for t in r["tasks"]:
            groups[pr[t["id"]]].append((t["split"], tuple(t["pool_correct"]), tuple(t["pool_completion_tokens"])))
        dup = [g for g in groups.values() if len(g) > 1]
        same = sum(len({x[1:] for x in g}) == 1 for g in dup)
        cross = sum(len({x[0] for x in g}) == 2 for g in dup)
        mism = [t for t in r["tasks"] if t["arch"]["seq_r1"]["tokens"] != t["pool_tokens"][0]
                or t["arch"]["seq_r1"]["acc"] != float(t["pool_correct"][0])]
        print(f"{m}: {len(groups)} unique prompts in {len(r['tasks'])} tasks; {len(dup)} duplicate groups "
              f"({cross} span calib+test, {same} with byte-identical pools); "
              f"seq_r1!=single on {len(mism)} tasks, {sum(cnt[pr[t['id']]] > 1 for t in mism)} of them duplicated")


def c_case_study():
    head("C1-C4  citation case study")
    cs = json.load(open(RES / "citation_case_study.json"))
    print(f"topics with >=1 'real' candidate: {round(cs['coverage'] * cs['n_topics'])}/{cs['n_topics']}; "
          f"G={cs['G_pairwise']:+.2f}; registry calls: {cs['registry_by_source']}")
    cache = json.load(open(RES / "openalex_cache.json"))
    for k, v in cache.items():
        if v.get("ratio") and v["ratio"] >= 0.88:
            print(f"  {v['ratio']:.2f} {v['source']:8s} '{k[:60]}' -> '{(v['title'] or '')[:60]}'")


if __name__ == "__main__":
    v1_truncation(); v1_v2_headline(); v3_winners_curse(); v4_clamping(); v6_snippet()
    s1_prediction(); s2_s3_gap(); s5_r3_duplicates(); c_case_study()
