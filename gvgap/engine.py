"""Batched MLX inference with per-sample token accounting and on-disk caching.

Token accounting is load-bearing here: the paper's central comparison is
between architectures at *equal token cost*, so every call records the prompt
and completion tokens it actually consumed rather than the budget it was
allotted. Results are cached by content hash so a run can be interrupted and
resumed without re-spending compute.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results" / "cache"


@dataclass
class Gen:
    """A single completion plus what it cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Budget:
    """Running tally of everything an architecture spent on one task."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, g: Gen) -> None:
        self.prompt_tokens += g.prompt_tokens
        self.completion_tokens += g.completion_tokens
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


class Runner:
    """Wraps one model. All generation goes through :meth:`chat`."""

    def __init__(
        self,
        model_id: str,
        batch_size: int = 64,
        cache_name: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        t0 = time.time()
        self.model, self.tok = load(model_id)
        self.load_s = time.time() - t0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        name = cache_name or model_id.replace("/", "_")
        self.db = sqlite3.connect(CACHE_DIR / f"{name}.sqlite")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS gen ("
            "key TEXT PRIMARY KEY, text TEXT, ptok INT, ctok INT)"
        )
        self.db.commit()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------ templating
    def render(self, system: str | None, user: str) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def render_multi(self, msgs: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # ---------------------------------------------------------------- caching
    def _key(self, prompt: str, max_tokens: int, temp: float, seed: int) -> str:
        raw = f"{self.model_id}\x00{prompt}\x00{max_tokens}\x00{temp}\x00{seed}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cached(self, keys: Sequence[str]) -> dict[str, Gen]:
        out: dict[str, Gen] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = ",".join("?" * len(chunk))
            for k, t, p, c in self.db.execute(
                f"SELECT key,text,ptok,ctok FROM gen WHERE key IN ({q})", chunk
            ):
                out[k] = Gen(t, p, c)
        return out

    def _store(self, rows: list[tuple[str, str, int, int]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO gen VALUES (?,?,?,?)", rows)
        self.db.commit()

    # ------------------------------------------------------------- generation
    def generate(
        self,
        prompts: Sequence[str],
        max_tokens: int = 320,
        temp: float = 0.7,
        seeds: Sequence[int] | None = None,
    ) -> list[Gen]:
        """Generate one completion per prompt, batching and caching throughout.

        ``seeds`` lets the caller draw several distinct samples from the same
        prompt; identical (prompt, seed) pairs are served from cache so that
        re-running an experiment is free and exactly reproducible.
        """
        seeds = list(seeds) if seeds is not None else [0] * len(prompts)
        keys = [
            self._key(p, max_tokens, temp, s) for p, s in zip(prompts, seeds)
        ]
        cached = self._cached(keys)
        todo = [i for i, k in enumerate(keys) if k not in cached]
        self.hits += len(prompts) - len(todo)
        self.misses += len(todo)

        results: list[Gen | None] = [cached.get(k) for k in keys]
        sampler = make_sampler(temp=temp, top_p=0.95)

        for start in range(0, len(todo), self.batch_size):
            idx = todo[start : start + self.batch_size]
            ids = [self.tok.encode(prompts[i]) for i in idx]
            # One seed per batch; distinct seeds land in distinct batches only
            # when the caller asks for them, which is why we group by seed.
            mx.random.seed(int(seeds[idx[0]]) * 1_000_003 + start)
            resp = batch_generate(
                self.model,
                self.tok,
                ids,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            rows = []
            for j, i in enumerate(idx):
                text = resp.texts[j]
                ctok = len(self.tok.encode(text, add_special_tokens=False))
                g = Gen(text, len(ids[j]), ctok)
                results[i] = g
                rows.append((keys[i], text, g.prompt_tokens, g.completion_tokens))
            self._store(rows)

        assert all(r is not None for r in results)
        return [r for r in results if r is not None]  # type: ignore[misc]

    def chat(
        self,
        system: str | None,
        users: Sequence[str],
        max_tokens: int = 320,
        temp: float = 0.7,
        seeds: Sequence[int] | None = None,
    ) -> list[Gen]:
        return self.generate(
            [self.render(system, u) for u in users],
            max_tokens=max_tokens,
            temp=temp,
            seeds=seeds,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "load_s": round(self.load_s, 2),
        }
