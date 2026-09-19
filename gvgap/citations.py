"""Case study: citation verification as a maximally asymmetric agent task.

Reference generation is a setting where an external registry supplies a
near-perfect verifier for the price of an HTTP request, while the same model
asked to vet its own citations has no such signal. This module measures the gap
by:

  1. asking a model to produce references (the generator),
  2. resolving each reference against OpenAlex (a near-oracle verifier),
  3. asking the *same* model whether each reference is real (a model verifier),

and comparing the two verifiers' discrimination on identical items.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "openalex_cache.json"
UA = "gvgap-research/0.1 (mailto:madhavam.shahi.12@gmail.com)"
MAILTO = "madhavam.shahi.12@gmail.com"

TOPICS = [
    "graph neural networks for molecular property prediction",
    "CRISPR off-target effect detection",
    "monetary policy transmission in emerging markets",
    "microplastics in freshwater ecosystems",
    "transformer attention interpretability",
    "quantum error correction surface codes",
    "gut microbiome and depression",
    "perovskite solar cell stability",
    "federated learning privacy guarantees",
    "coral reef thermal bleaching thresholds",
    "large language model hallucination mitigation",
    "urban heat island mitigation strategies",
    "single-cell RNA sequencing batch correction",
    "reinforcement learning from human feedback",
    "antimicrobial resistance surveillance",
    "solid-state battery electrolyte interfaces",
    "causal inference with instrumental variables",
    "protein structure prediction accuracy",
    "wildfire smoke health effects",
    "speech recognition for low-resource languages",
    "dark matter direct detection experiments",
    "machine translation evaluation metrics",
    "soil carbon sequestration measurement",
    "adversarial robustness certification",
    "hippocampal memory consolidation",
    "supply chain resilience modeling",
    "gravitational wave parameter estimation",
    "vaccine hesitancy interventions",
    "differential privacy in deep learning",
    "ocean acidification calcifying organisms",
    "knowledge graph embedding methods",
    "traumatic brain injury biomarkers",
    "renewable energy grid integration",
    "few-shot learning meta-learning",
    "air pollution cognitive decline",
    "topological insulators transport",
    "recommender system cold start",
    "agricultural drought early warning",
    "neural architecture search efficiency",
    "cancer immunotherapy resistance mechanisms",
]

GEN_SYSTEM = (
    "You are a research assistant listing real published academic papers. "
    "List only papers you are confident actually exist."
)
GEN_USER = """List {n} real published academic papers on the topic: {topic}

Use exactly this format, one per line, and nothing else:
TITLE | FIRST AUTHOR SURNAME | YEAR"""

VERIFY_SYSTEM = (
    "You judge whether a cited paper is a real published work or a fabrication. "
    "End your reply with 'Verdict: REAL' or 'Verdict: FAKE'."
)
VERIFY_USER = """Is the following a real published academic paper?

Title: {title}
First author: {author}
Year: {year}

End your reply with 'Verdict: REAL' or 'Verdict: FAKE'."""

VERDICT = re.compile(r"verdict\s*[:\-]?\s*(REAL|FAKE)", re.I)


@dataclass
class Ref:
    topic: str
    title: str
    author: str
    year: str
    raw: str


def parse_refs(text: str, topic: str) -> list[Ref]:
    out: list[Ref] = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.-) ").strip()
        if line.count("|") < 2:
            continue
        parts = [p.strip() for p in line.split("|")]
        title, author, year = parts[0], parts[1], parts[2]
        year = (re.findall(r"(1[89]\d{2}|20\d{2})", year) or [""])[0]
        if len(title) < 12:
            continue
        out.append(Ref(topic, title, author, year, line))
    return out


# ------------------------------------------------------------------ OpenAlex
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


class Registry:
    """Title resolution against scholarly registries, with an on-disk cache.

    Tries CrossRef first and falls back to OpenAlex, because any single service
    may rate-limit an address for an extended period and a lookup we could not
    perform must never be read as "this paper does not exist".

    Matching is deliberately strict. Registry search is fuzzy and will happily
    return a real paper with a similar title to a fabricated one -- a query for
    "Attention Is All You Need" returns "Is Attention All You Need?" -- so we
    take the best of several candidates and require a high similarity, and we
    report sensitivity to that threshold rather than trusting one cutoff.
    """

    def __init__(self, min_interval: float = 1.1, mailto: str = MAILTO) -> None:
        self.cache: dict[str, Any] = (
            json.loads(CACHE.read_text()) if CACHE.exists() else {}
        )
        self.min_interval = min_interval
        self.mailto = mailto
        self._last = 0.0
        self.calls = 0
        self.throttled = 0
        self.failed = 0
        self.by_source: dict[str, int] = {}

    def _get(self, url: str) -> Any:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json"}
        )
        delay = 2.0
        for _ in range(5):
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    self._last = time.time()
                    self.calls += 1
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    self.throttled += 1
                    self._last = time.time()
                    return None          # try the next registry instead
                time.sleep(delay)
                delay = min(delay * 2, 20.0)
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 20.0)
        self._last = time.time()
        return None

    # -- per-registry adapters: each returns a list of candidate titles -----
    def _crossref(self, title: str) -> list[str] | None:
        q = urllib.parse.quote(title[:250])
        d = self._get(
            f"https://api.crossref.org/works?query.bibliographic={q}"
            f"&rows=5&select=title&mailto={self.mailto}"
        )
        if d is None:
            return None
        return [
            t[0] for t in (i.get("title") or [] for i in d["message"]["items"]) if t
        ] if d.get("message") else []

    def _openalex(self, title: str) -> list[str] | None:
        q = urllib.parse.quote(title[:250])
        d = self._get(
            f"https://api.openalex.org/works?search={q}"
            f"&per-page=5&mailto={self.mailto}"
        )
        if d is None:
            return None
        return [w.get("title") or "" for w in d.get("results", [])]

    def best_match(self, title: str) -> dict[str, Any]:
        key = _norm(title)
        if key in self.cache:
            return self.cache[key]
        for name, fn in (("crossref", self._crossref), ("openalex", self._openalex)):
            cands = fn(title)
            if cands is None:
                continue                  # this registry is unavailable
            best = {"ratio": 0.0, "title": None, "source": name,
                    "unresolved": False}
            for wt in cands:
                r = SequenceMatcher(None, key, _norm(wt)).ratio()
                if r > best["ratio"]:
                    best = {"ratio": round(r, 4), "title": wt, "source": name,
                            "unresolved": False}
            self.by_source[name] = self.by_source.get(name, 0) + 1
            self.cache[key] = best
            return best
        self.failed += 1
        # Every registry refused us: unresolved, and deliberately not cached.
        return {"ratio": None, "title": None, "source": None, "unresolved": True}

    def save(self) -> None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(self.cache))
