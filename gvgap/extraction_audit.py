"""Audit answer extraction: how often does the grader fail to find any answer?

A brittle extractor silently depresses every accuracy number in the paper, so
we measure the failure rate instead of assuming it is zero. Cache keys are
deterministic in (model, prompt, max tokens, temperature, seed), so this
re-grades the stored pool generations without running any inference.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgap import prompts
from gvgap.run import MAX_TOKENS, POOL_N
from gvgap.verify import check

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
CACHE = ROOT / "results" / "cache"
FAIL_CODES = {"no_number", "no_choice", "no_name", "no_code", "empty",
              "unparseable"}
FULL_NAME = {
    "Qwen_Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}


def key(model: str, prompt: str, max_tokens: int, temp: float, seed: int) -> str:
    raw = f"{model}\x00{prompt}\x00{max_tokens}\x00{temp}\x00{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()


def main() -> None:
    from mlx_lm import load

    rows = []
    for model_dir in sorted(RAW.glob("*")):
        full = FULL_NAME.get(model_dir.name)
        db = CACHE / f"{model_dir.name}.sqlite"
        if not full or not db.exists():
            continue
        _, tok = load(full)
        con = sqlite3.connect(db)
        texts = dict(con.execute("select key,text from gen"))
        for fam_file in sorted(model_dir.glob("*.json")):
            fam = fam_file.stem
            items = [json.loads(l) for l in
                     (ROOT / "data" / f"{fam}.jsonl").open()]
            ids = {t["id"] for t in json.loads(fam_file.read_text())["tasks"]}
            items = [it for it in items if it["id"] in ids]
            mt = MAX_TOKENS[fam]
            codes: Counter = Counter()
            total = missing = 0
            for it in items:
                rendered = tok.apply_chat_template(
                    [{"role": "system", "content": prompts.SYSTEM[fam]},
                     {"role": "user", "content": it["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
                for s in range(POOL_N):
                    t = texts.get(key(full, rendered, mt, 0.7, s))
                    if t is None:
                        missing += 1
                        continue
                    total += 1
                    c = check(t, it)
                    if c.detail in FAIL_CODES or c.extracted is None:
                        codes[c.detail or "none"] += 1
            fails = sum(codes.values())
            rows.append({
                "model": model_dir.name.replace("Qwen_Qwen2.5-", "").replace("-Instruct", ""),
                "family": fam, "n_graded": total, "n_cache_miss": missing,
                "extraction_failures": fails,
                "failure_rate": round(fails / total, 4) if total else None,
                "codes": dict(codes),
            })
            print(f"{rows[-1]['model']:5s} {fam:8s} n={total:5d} "
                  f"fail={fails:4d} ({rows[-1]['failure_rate']:.3f}) {dict(codes)}",
                  flush=True)
    (ROOT / "results" / "extraction_audit.json").write_text(
        json.dumps(rows, indent=1))
    if rows:
        worst = max(rows, key=lambda r: r["failure_rate"] or 0)
        print(f"\nworst cell: {worst['model']}/{worst['family']} "
              f"{worst['failure_rate']:.3f}")


if __name__ == "__main__":
    main()
