"""Assemble the six task families into a single frozen task suite.

Every family is subsampled with the same seed and written to data/*.jsonl so
the exact item set used in the paper is redistributable and re-derivable.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap import synth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 20260919
N_PER_FAMILY = 120


def _load_parquet(repo: str, filename: str) -> pd.DataFrame:
    path = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    return pd.read_parquet(path)


def build_gsm8k(n: int) -> list[dict]:
    df = _load_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet")
    df = df.sample(n=n, random_state=SEED).reset_index(drop=True)
    out = []
    for _, r in df.iterrows():
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append(
            {
                "family": "gsm8k",
                "prompt": r["question"].strip(),
                "answer": gold,
                "meta": {},
            }
        )
    return out


def build_arc(n: int) -> list[dict]:
    df = _load_parquet("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        ch = r["choices"]
        labels = list(ch["label"])
        texts = list(ch["text"])
        # A handful of ARC items use numeric labels; normalise to letters.
        if labels and labels[0] not in "ABCDE":
            labels = [chr(ord("A") + i) for i in range(len(labels))]
        if r["answerKey"] not in labels and r["answerKey"].isdigit():
            key = chr(ord("A") + int(r["answerKey"]) - 1)
        else:
            key = r["answerKey"]
        if key not in labels or len(labels) < 4:
            continue
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        rows.append(
            {
                "family": "arc",
                "prompt": f"{r['question'].strip()}\n{opts}",
                "answer": key,
                "meta": {"n_choices": len(labels)},
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


def build_mbpp(n: int) -> list[dict]:
    df = _load_parquet("google-research-datasets/mbpp", "full/test-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        tests = [t for t in list(r["test_list"]) if t.strip()]
        if not tests:
            continue
        rows.append(
            {
                "family": "mbpp",
                "prompt": (
                    f"{r['text'].strip()}\n"
                    f"Your solution must satisfy this test:\n{tests[0]}"
                ),
                "answer": "",
                "meta": {"tests": tests, "task_id": int(r["task_id"])},
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


BUILDERS = {
    "gsm8k": build_gsm8k,
    "arc": build_arc,
    "mbpp": build_mbpp,
    "csp": lambda n: synth.gen_csp(n, seed=SEED),
    "logic": lambda n: synth.gen_logic(n, seed=SEED),
    "wordcon": lambda n: synth.gen_wordcon(n, seed=SEED),
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    summary = {}
    for fam, fn in BUILDERS.items():
        items = fn(N_PER_FAMILY)
        for i, it in enumerate(items):
            it["id"] = f"{fam}-{i:04d}"
            # A fixed calibration/held-out split: H2 is a *prediction* claim, so
            # the split must be frozen before any model is run.
            it["split"] = "calib" if i < 30 else "test"
        path = DATA / f"{fam}.jsonl"
        with path.open("w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        summary[fam] = len(items)
        print(f"{fam:8s} {len(items):4d} -> {path.name}")
    (DATA / "suite.json").write_text(
        json.dumps({"seed": SEED, "n_per_family": N_PER_FAMILY, "counts": summary}, indent=2)
    )


if __name__ == "__main__":
    main()
