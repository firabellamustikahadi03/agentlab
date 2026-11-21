"""Hierarchical span tracer (OpenTelemetry-style, but tiny and dependency-free)."""
from __future__ import annotations

import contextlib
import itertools
import time
import uuid
from collections.abc import Iterator
from typing import Any

from .store import SqliteStore


class Span:
    def __init__(self, tracer: Tracer, kind: str, name: str, input: Any, parent_id: str | None):
        self.tracer = tracer
        self.d = {
            "span_id": uuid.uuid4().hex[:12], "run_id": tracer.run_id, "parent_id": parent_id,
            "kind": kind, "name": name, "input": input, "output": None, "status": "ok",
            "started": time.time(), "ended": None, "attrs": {}, "seq": next(tracer._seq),
        }

    @property
    def id(self) -> str:
        return self.d["span_id"]

    def set(self, **attrs: Any) -> None:
        self.d["attrs"].update(attrs)

    def finish(self, output: Any = None, status: str | None = None) -> None:
        if output is not None:
            self.d["output"] = output
        if status:
            self.d["status"] = status
        self.d["ended"] = time.time()
        self.tracer.store.save_span(self.d)


class Tracer:
    """One tracer per run. `kind` is one of: run, plan, step, llm, tool, critic."""

    def __init__(self, store: SqliteStore, run_id: str, start_seq: int = 0):
        self.store, self.run_id = store, run_id
        self._seq = itertools.count(start_seq)
        self._stack: list[Span] = []

    @contextlib.contextmanager
    def span(self, kind: str, name: str, input: Any = None) -> Iterator[Span]:
        parent = self._stack[-1].id if self._stack else None
        s = Span(self, kind, name, input, parent)
        self._stack.append(s)
        try:
            yield s
        except Exception as e:
            s.finish(output={"error": f"{type(e).__name__}: {e}"}, status="error")
            raise
        else:
            s.finish()
        finally:
            self._stack.pop()
