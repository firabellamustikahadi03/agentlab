"""Scorers: each check is a small dict spec; a task passes when ALL its checks pass."""
from __future__ import annotations

import re

from ..tools import run_python


def _num(text: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text or "")
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))  # last number in the answer is the final one
    except ValueError:
        return None


def check(spec: dict, answer: str | None, spans: list[dict]) -> tuple[bool, str]:
    t, a = spec["type"], (answer or "")
    if t == "exact":
        ok = a.strip().lower() == str(spec["value"]).strip().lower()
        return ok, f"exact {spec['value']!r}"
    if t == "contains":
        ok = str(spec["value"]).lower() in a.lower()
        return ok, f"contains {spec['value']!r}"
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
    if t == "python_assert":
        try:
            run_python(f"answer = {a!r}\n{spec['code']}")
            return True, "python_assert"
        except Exception as e:
            return False, f"python_assert failed: {e}"
    raise ValueError(f"unknown check type {t!r}")


def score(checks: list[dict], answer: str | None, spans: list[dict]) -> tuple[bool, float, str]:
    if not checks:
        return bool(answer), float(bool(answer)), "no checks"
    results = [check(c, answer, spans) for c in checks]
    passed = sum(ok for ok, _ in results)
    return passed == len(results), passed / len(results), "; ".join(
        ("PASS " if ok else "FAIL ") + d for ok, d in results
    )
