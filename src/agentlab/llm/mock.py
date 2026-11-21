"""Offline providers: ScriptedProvider (tests) and MockAgentProvider (demo / eval without keys)."""
from __future__ import annotations

import json
import re
import time

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


class MockAgentProvider:
    """A tiny deterministic 'model' that can solve arithmetic and simple python tasks.

    It reads the role marker ([ROLE:planner|executor|critic]) that the runtime puts in the
    system prompt and answers in the same JSON protocol a real model must follow.
    Lets the whole stack (runtime, tracing, eval) be exercised with zero API keys.
    """

    def __init__(self, name: str = "mock", model: str = "mock-agent-1", fail_first: bool = False):
        self.name, self.model = name, model
        self.fail_first = fail_first  # if True, executor answers wrongly once -> exercises critic/replan
        self._failed = False

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        system = messages[0].content
        user_first = messages[1].content if len(messages) > 1 else ""
        if "[ROLE:planner]" in system:
            out = self._plan(user_first)
        elif "[ROLE:critic]" in system:
            out = self._critic(user_first)
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

    def _execute(self, messages: list[Message]) -> dict:
        first = messages[1].content
        task = self._task_of(first)
        observations = [m.content for m in messages if m.role == "user" and m.content.startswith("OBSERVATION:")]
        if self.fail_first and not self._failed:
            self._failed = True
            return {"final": "wrong answer"}
        if observations:
            return {"final": observations[-1].removeprefix("OBSERVATION:").strip()}
        code = re.search(r"```python\n(.*?)```", task, re.S)
        if code:
            return {"action": {"tool": "python", "args": {"code": code.group(1)}}}
        expr = _EXPR.search(task)
        if expr:
            return {"action": {"tool": "calculator", "args": {"expression": expr.group(1).strip()}}}
        return {"final": "I do not know."}
