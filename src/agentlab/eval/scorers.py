"""Scorers: each check is a small dict spec; a task passes when ALL its checks pass."""
from __future__ import annotations

import re

from .. import prompts
from ..jsonutil import extract_json
from ..llm.base import Message, Provider
from ..tools import run_python


def _num(text: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text or "")
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))  # last number in the answer is the final one
    except ValueError:
        return None


def _norm_score(v: object) -> float | None:
    """Accept 0..1, 0..10 or 0..100 scales (small models love to answer '8/10')."""
    try:
        x = float(str(v).split("/")[0]) if isinstance(v, str) else float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if x > 100 or x < 0:
        return None
    if x > 10:
        x /= 100
    elif x > 1:
        x /= 10
    return x


def judge_answer(judge: Provider, task: str, answer: str, rubric: str) -> tuple[float, str]:
    """Ask a judge model for a 0..1 score. Never raises: a broken judge scores 0 with the reason."""
    msgs = [
        Message("system", prompts.JUDGE),
        Message("user", f"TASK: {task}\nRUBRIC: {rubric}\nANSWER: {answer}"),
    ]
    try:
        text = judge.complete(msgs, temperature=0.0).content
        obj = extract_json(text)
        if obj is None:
            m = re.search(r"score\D{0,12}(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", text, re.I)
            obj = {"score": m.group(1).replace(" ", "")} if m else None
        if obj is None:  # one repair attempt, like the runtime does
            msgs += [Message("assistant", text), Message("user", "Reply again with ONLY the JSON object.")]
            obj = extract_json(judge.complete(msgs, temperature=0.0).content)
    except Exception as e:  # network errors, exhausted scripted providers, ...
        return 0.0, f"judge error: {type(e).__name__}: {e}"
    sc = _norm_score((obj or {}).get("score"))
    if sc is None:
        return 0.0, "unparseable judge verdict"
    return sc, str((obj or {}).get("reason", ""))[:200]


def check(
    spec: dict, answer: str | None, spans: list[dict], judge: Provider | None = None, task: str = ""
) -> tuple[bool, str]:
    t, a = spec["type"], (answer or "")
    if t == "exact":
        ok = a.strip().lower() == str(spec["value"]).strip().lower()
        return ok, f"exact {spec['value']!r}"
    if t == "contains":
        ok = str(spec["value"]).lower() in a.lower()
        return ok, f"contains {spec['value']!r}"
    if t == "not_contains":
        ok = str(spec["value"]).lower() not in a.lower()
        return ok, f"not_contains {spec['value']!r}"
    if t == "regex":
        return re.search(spec["pattern"], a, re.S) is not None, f"regex {spec['pattern']!r}"
    if t == "numeric":
        n = _num(a)
        tol = spec.get("tol", 1e-6)
        ok = n is not None and abs(n - float(spec["value"])) <= tol * max(1.0, abs(float(spec["value"])))
        return ok, f"numeric {spec['value']} (got {n})"
    if t == "tool_used":
        used = {s["name"] for s in spans if s["kind"] == "tool"}
        return spec["tool"] in used, f"tool_used {spec['tool']} (used {sorted(used)})"
    if t == "tool_not_used":
        used = {s["name"] for s in spans if s["kind"] == "tool"}
        return spec["tool"] not in used, f"tool_not_used {spec['tool']} (used {sorted(used)})"
    if t == "recovered_from_error":  # some tool call failed, and the run still produced an answer
        errs = sum(1 for s in spans if s["kind"] == "tool" and s["status"] == "error")
        return errs > 0 and bool(a.strip()), f"recovered_from_error ({errs} tool errors)"
    if t == "judge":
        if judge is None:
            return False, "judge check needs a judge provider (--judge)"
        sc, why = judge_answer(judge, task, a, spec["rubric"])
        return sc >= spec.get("threshold", 0.5), f"judge {sc:.2f} ({why})"
    if t == "python_assert":
        try:
            run_python(f"answer = {a!r}\n{spec['code']}")
            return True, "python_assert"
        except Exception as e:
            return False, f"python_assert failed: {e}"
    raise ValueError(f"unknown check type {t!r}")


def score(
    checks: list[dict], answer: str | None, spans: list[dict], judge: Provider | None = None, task: str = ""
) -> tuple[bool, float, str]:
    if not checks:
        return bool(answer), float(bool(answer)), "no checks"
    results = [check(c, answer, spans, judge, task) for c in checks]
    passed = sum(ok for ok, _ in results)
    return passed == len(results), passed / len(results), "; ".join(
        ("PASS " if ok else "FAIL ") + d for ok, d in results
    )
