"""Deterministic replay: re-run a recorded run using its recorded LLM outputs."""
from __future__ import annotations

from ..llm.base import LLMResponse, Message, Usage
from .store import SqliteStore


class ReplayProvider:
    """Serves the recorded LLM completions of a past run, in order.

    Re-running the runtime with this provider reproduces the exact agent behaviour
    (tool calls are re-executed), so bugs in tools/runtime logic can be debugged without
    spending tokens or hitting rate limits. Divergence is detected and reported.
    """

    def __init__(self, store: SqliteStore, run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise KeyError(f"run {run_id} not found")
        self.name, self.model = f"replay:{run_id}", run.get("model") or "replay"
        self._llm = [s for s in store.get_spans(run_id) if s["kind"] == "llm"]
        self._i = 0
        self.divergences: list[str] = []

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse:
        if self._i >= len(self._llm):
            raise RuntimeError("replay exhausted: run asked for more LLM calls than were recorded")
        rec = self._llm[self._i]
        self._i += 1
        recorded_n = (rec["input"] or {}).get("n_messages")
        if recorded_n is not None and recorded_n != len(messages):
            self.divergences.append(f"call {self._i}: recorded {recorded_n} messages, got {len(messages)}")
        a = rec["attrs"] or {}
        return LLMResponse(
            content=(rec["output"] or {}).get("content", ""),
            usage=Usage(a.get("prompt_tokens", 0), a.get("completion_tokens", 0)),
            model=self.model,
        )
