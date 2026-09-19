"""Canonical answer extraction, used for majority voting and clustering.

Returns ``None`` when a family has no short canonical answer, which is itself a
finding: verifier-free aggregation such as self-consistency simply does not
apply to open-ended outputs.
"""
from __future__ import annotations

import re
from typing import Any

from gvgap.verify import BOXED, CHOICE, NUMBER, ANSWER_TAG, WORD, _numeric_answer


def canonical(text: str, item: dict[str, Any]) -> str | None:
    fam = item["family"]
    if fam in ("gsm8k", "csp"):
        v = _numeric_answer(text)
        if v is None:
            return None
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return v
    if fam == "logic":
        cands = item["meta"]["candidates"]
        hits = [(text.rfind(c), c) for c in cands if c in text]
        return max(hits)[1] if hits else None
    if fam == "arc":
        m = BOXED.findall(text)
        if m:
            c = CHOICE.findall(m[-1].upper())
            if c:
                return c[-1]
        tail = ANSWER_TAG.split(text)
        if len(tail) > 1:
            c = CHOICE.findall(tail[-1].upper())
            if c:
                return c[0]
        c = CHOICE.findall(text.upper())
        return c[-1] if c else None
    # wordcon and mbpp have no canonical short form.
    return None


HAS_CANONICAL = {"gsm8k", "csp", "logic", "arc"}
