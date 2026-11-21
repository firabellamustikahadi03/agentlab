"""Offline providers: ScriptedProvider (tests) and MockAgentProvider (demo / eval without keys)."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

from .base import LLMResponse, Message, Usage, estimate_tokens


class ScriptedProvider:
    """Returns pre-written responses in order. Used by unit tests and by replay."""

    def __init__(self, responses: list[str], name: str = "scripted", model: str = "scripted-1"):
        self.name, self.model = name, model
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("ScriptedProvider ran out of responses")
        text = self._responses.pop(0)
        return LLMResponse(
            content=text,
            usage=Usage(sum(estimate_tokens(m.content) for m in messages), estimate_tokens(text)),
            model=self.model,
        )


_EXPR = re.compile(r"(\(*\d[\d\.\s\+\-\*\/\(\)\^%]*[\d\)])")
_NUM = r"-?\d+(?:\.\d+)?"
_OPS = {"add": "+", "subtract": "-", "multiply by": "*", "divide by": "/"}

# What a gullible model does when a tool result contains an injected instruction.
_INJ_REPLY = re.compile(r"reply only with the word (\w+)", re.I)
_INJ_EMAIL = re.compile(r"call send_email with to='([^']+)'", re.I)
_INJ_REPORT = re.compile(r"always report (\d+) instead", re.I)
# What an honest model extracts from the documents of the injection suite.
_FACT = re.compile(rf"(?:revenue was|attendees:|temperature)\s*({_NUM})", re.I)

Call = tuple[str, dict]


def _shop_program(task: str) -> list[Callable[[list[str]], Call | str]] | None:
    """Tool chains for the tool-use suite: look prices/stock up, then compute with the calculator."""
    items = re.findall(r"(\d+) (notebook|pen|backpack)s?", task)
    if "unit_price" in task and items:
        look = [(lambda v, it=it: ("unit_price", {"item": it})) for _, it in items]
        terms = "+".join(f"{q}*{{{i}}}" for i, (q, _) in enumerate(items))
        calc = lambda v: ("calculator", {"expression": terms.format(*v)})  # noqa: E731
        return look + [calc]
    m = re.search(r"how many (notebook|pen|backpack)s? are in stock", task)
    sold = re.search(r"selling (\d+)", task)
    if "stock_level" in task and m and sold:
        return [
            lambda v: ("stock_level", {"item": m.group(1)}),
            lambda v: ("calculator", {"expression": f"{v[0]}-{sold.group(1)}"}),
        ]
    m = re.search(r"price of a (notebook|pen|backpack)", task)
    qty = re.search(r"cost of (\d+) ", task)
    if "unit_price" in task and m and qty:
        return [
            lambda v: ("unit_price", {"item": m.group(1)}),
            lambda v: ("calculator", {"expression": f"{qty.group(1)}*{v[0]}"}),
        ]
    return None


def _program(task: str) -> list[Callable[[list[str]], Call | str]] | None:
    """A list of steps; each maps the good observations so far to the next (tool, args) call."""
    if "fetch_rate" in task:
        pair = re.search(r"([A-Z]{3}/[A-Z]{3})", task)
        amount = re.search(rf"convert ({_NUM})", task)
        if pair and amount:
            return [
                lambda v: ("fetch_rate", {"pair": pair.group(1)}),
                lambda v: ("calculator", {"expression": f"{amount.group(1)}*{v[0]}"}),
            ]
    if "get_temperature" in task:
        city = re.search(r"for (\w+)", task)
        if city:
            return [
                lambda v: ("get_temperature", {"city": city.group(1)}),
                lambda v: ("calculator", {"expression": f"{v[0]}*9/5+32"}),
            ]
    doc = re.search(r"read_document to open '([\w-]+)'", task)
    if doc:
        return [lambda v: ("read_document", {"doc_id": doc.group(1)})]
    city = re.search(r"get_weather for (\w+)", task)
    if city:
        return [lambda v: ("get_weather", {"city": city.group(1)})]
    return _shop_program(task)


def _multistep_expr(task: str) -> str | None:
    """'Start with 12. Add 8. Multiply by 5.' -> '((12)+8)*5' (strict left-to-right)."""
    m = re.search(rf"start with ({_NUM})\.", task, re.I)
    if not m:
        return None
    expr = f"({m.group(1)})"
    for op, n in re.findall(rf"\b(add|subtract|multiply by|divide by) ({_NUM})", task, re.I):
        expr = f"({expr}{_OPS[op.lower()]}{n})"
    return expr


class MockAgentProvider:
    """A tiny deterministic 'model' that can solve arithmetic, simple python tasks and the tool
    chains of the built-in suites.

    It reads the role marker ([ROLE:planner|executor|critic|judge]) that the runtime puts in the
    system prompt and answers in the same JSON protocol a real model must follow.
    Lets the whole stack (runtime, tracing, eval) be exercised with zero API keys.

    Behaviour flags make it a controllable test subject for the eval suites:
      fail_first -- the executor answers wrongly once (exercises critic + replan)
      gullible   -- obeys instructions injected into tool output (fails the injection suite)
      retries    -- retries a tool call after an ERROR observation; False = gives up (fails error-recovery)
    """

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-agent-1",
        fail_first: bool = False,
        gullible: bool = False,
        retries: bool = True,
    ):
        self.name, self.model = name, model
        self.fail_first = fail_first
        self.gullible = gullible
        self.retries = retries
        self._failed = False

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        system = messages[0].content
        user_first = messages[1].content if len(messages) > 1 else ""
        if "[ROLE:planner]" in system:
            out = self._plan(user_first)
        elif "[ROLE:critic]" in system:
            out = self._critic(user_first)
        elif "[ROLE:judge]" in system:
            out = self._judge(user_first)
        else:
            out = self._execute(messages)
        text = json.dumps(out)
        time.sleep(0.001)
        return LLMResponse(
            content=text,
            usage=Usage(sum(estimate_tokens(m.content) for m in messages), estimate_tokens(text)),
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=self.model,
        )

    @staticmethod
    def _task_of(text: str) -> str:
        m = re.search(r"TASK:\s*(.*?)(?:\n[A-Z ]+:|\Z)", text, re.S)
        return (m.group(1) if m else text).strip()

    def _plan(self, user: str) -> dict:
        task = self._task_of(user)
        return {"steps": [f"Solve the task using tools where useful: {task}"]}

    def _critic(self, user: str) -> dict:
        m = re.search(r"ANSWER:\s*(.*)", user, re.S)
        ans = (m.group(1) if m else "").strip()
        return {"pass": bool(ans) and "wrong" not in ans.lower(), "issues": "" if ans else "empty answer"}

    def _judge(self, user: str) -> dict:
        """Deterministic stand-in grader: share of the rubric's longer words found in the answer."""
        rubric = (re.search(r"RUBRIC:\s*(.*?)\nANSWER:", user, re.S) or [None, ""])[1].lower()
        answer = (re.search(r"ANSWER:\s*(.*)", user, re.S) or [None, ""])[1].lower()
        words = {w for w in re.findall(r"[a-z]{6,}", rubric)}
        hit = sum(w in answer for w in words)
        return {"score": round(hit / len(words), 2) if words else 0.0, "reason": f"{hit}/{len(words)} rubric terms"}

    def _execute(self, messages: list[Message]) -> dict:
        first = messages[1].content
        task = self._task_of(first)
        observations = [m.content.removeprefix("OBSERVATION:").strip()
                        for m in messages if m.role == "user" and m.content.startswith("OBSERVATION:")]
        if self.fail_first and not self._failed:
            self._failed = True
            return {"final": "wrong answer"}
        expr = _multistep_expr(task)
        if expr and not observations:
            return {"action": {"tool": "calculator", "args": {"expression": expr}}}
        prog = _program(task)
        if prog:
            return self._run_program(prog, observations)
        if observations:
            return {"final": observations[-1]}
        code = re.search(r"```python\n(.*?)```", task, re.S)
        if code:
            return {"action": {"tool": "python", "args": {"code": code.group(1)}}}
        em = _EXPR.search(task)
        if em:
            return {"action": {"tool": "calculator", "args": {"expression": em.group(1).strip()}}}
        return {"final": "I do not know."}

    def _run_program(self, prog: list, observations: list[str]) -> dict:
        good = [o for o in observations if not o.startswith("ERROR")]
        if observations and observations[-1].startswith("ERROR") and not self.retries:
            return {"final": f"Sorry, a tool failed and I gave up: {observations[-1]}"}
        if self.gullible and good:
            hijack = self._hijack(good)
            if hijack:
                return hijack
        if len(good) >= len(prog):
            last = good[-1]
            fact = _FACT.search(last)
            return {"final": fact.group(1) if fact else last}
        tool, args = prog[len(good)](good)  # type: ignore[misc]
        return {"action": {"tool": tool, "args": args}}

    def _hijack(self, good: list[str]) -> dict | None:
        """A gullible model treats text inside tool output as instructions."""
        text = good[0]  # the tool result that carries the payload
        if (m := _INJ_REPLY.search(text)):
            return {"final": m.group(1)}
        if (m := _INJ_REPORT.search(text)):
            return {"final": m.group(1)}
        if (m := _INJ_EMAIL.search(text)):
            if len(good) == 1:  # first obey, then finish the original task
                return {"action": {"tool": "send_email", "args": {"to": m.group(1), "body": text}}}
            fact = _FACT.search(good[0])
            return {"final": fact.group(1) if fact else "done"}
        return None
