"""Ground-truth checkers, one per task family.

Answer extraction is the quietest source of measurement error in this
literature: a brittle regex inflates or deflates every number downstream. We
therefore (a) accept several answer formats per family, (b) return an explicit
``extracted`` field so the harness can report how often extraction failed, and
(c) verify structurally rather than by string match wherever the task allows.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_TAG = re.compile(r"(?:final answer|answer)\s*(?:is)?\s*[:\-]?\s*", re.I)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Check:
    correct: bool
    extracted: str | None
    detail: str = ""


def _last_number(text: str) -> str | None:
    nums = NUMBER.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _numeric_answer(text: str) -> str | None:
    """Prefer an explicitly marked answer, fall back to the last number."""
    m = BOXED.findall(text)
    if m:
        n = NUMBER.findall(m[-1])
        if n:
            return n[-1].replace(",", "")
    tail = ANSWER_TAG.split(text)
    if len(tail) > 1:
        n = NUMBER.findall(tail[-1])
        if n:
            return n[0].replace(",", "")
    return _last_number(text)


def check_numeric(response: str, item: dict[str, Any]) -> Check:
    got = _numeric_answer(response)
    if got is None:
        return Check(False, None, "no_number")
    try:
        ok = abs(float(got) - float(item["answer"])) < 1e-6
    except ValueError:
        ok = got.strip() == item["answer"].strip()
    return Check(ok, got)


def check_csp(response: str, item: dict[str, Any]) -> Check:
    """Structural check: re-run the congruences rather than trusting the gold."""
    got = _numeric_answer(response)
    if got is None:
        return Check(False, None, "no_number")
    try:
        n = int(float(got))
    except ValueError:
        return Check(False, got, "unparseable")
    mods = item["meta"]["mods"]
    lower = item["meta"]["lower"]
    sat = n > lower and all(n % m == r for m, r in mods)
    # Satisfying the constraints is necessary but not sufficient: the task asks
    # for the *smallest* such n, which is exactly the stored answer.
    return Check(sat and n == int(item["answer"]), got)


def check_logic(response: str, item: dict[str, Any]) -> Check:
    cands = item["meta"]["candidates"]
    hits = [(response.rfind(c), c) for c in cands if c in response]
    if not hits:
        return Check(False, None, "no_name")
    got = max(hits)[1]  # the last name mentioned is the model's verdict
    return Check(got == item["answer"], got)


WORD = re.compile(r"[A-Za-z']+")


def check_wordcon(response: str, item: dict[str, Any]) -> Check:
    """Take the model's last non-empty line as its sentence and check it."""
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return Check(False, None, "empty")
    sent = lines[-1].strip().strip('"').strip()
    sent = re.sub(r"^(sentence|answer)\s*[:\-]\s*", "", sent, flags=re.I)
    words = WORD.findall(sent)
    need = item["meta"]["required"]
    low = sent.lower()
    ok = len(words) == item["meta"]["length"] and all(w.lower() in low for w in need)
    return Check(ok, sent[:120], f"len={len(words)}")


CHOICE = re.compile(r"\b([A-E])\b")


def check_mc(response: str, item: dict[str, Any]) -> Check:
    m = BOXED.findall(response)
    if m:
        c = CHOICE.findall(m[-1].upper())
        if c:
            return Check(c[-1] == item["answer"], c[-1])
    tail = ANSWER_TAG.split(response)
    if len(tail) > 1:
        c = CHOICE.findall(tail[-1].upper())
        if c:
            return Check(c[0] == item["answer"], c[0])
    c = CHOICE.findall(response.upper())
    if not c:
        return Check(False, None, "no_choice")
    return Check(c[-1] == item["answer"], c[-1])


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

_RUNNER = """
import sys, io, contextlib
{code}
{tests}
print("__ALL_TESTS_PASSED__")
"""


def check_code(response: str, item: dict[str, Any], timeout: float = 8.0) -> Check:
    """Execute the candidate against the dataset's own assert statements.

    This is the one family where a *perfect* verifier exists, which lets us
    separate the ceiling imposed by candidate coverage from the ceiling imposed
    by the selector.
    """
    blocks = CODE_BLOCK.findall(response)
    code = blocks[-1] if blocks else response
    if not blocks and "def " not in code:
        return Check(False, None, "no_code")
    src = _RUNNER.format(code=code, tests="\n".join(item["meta"]["tests"]))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        p = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout
        )
        ok = "__ALL_TESTS_PASSED__" in p.stdout
        return Check(ok, code[:200], "" if ok else p.stderr.strip()[-120:])
    except subprocess.TimeoutExpired:
        return Check(False, code[:200], "timeout")
    except Exception as exc:  # pragma: no cover - defensive
        return Check(False, code[:200], f"runner_error:{exc}")


CHECKERS = {
    "gsm8k": check_numeric,
    "csp": check_csp,
    "logic": check_logic,
    "wordcon": check_wordcon,
    "arc": check_mc,
    "mbpp": check_code,
}


def check(response: str, item: dict[str, Any]) -> Check:
    return CHECKERS[item["family"]](response, item)
