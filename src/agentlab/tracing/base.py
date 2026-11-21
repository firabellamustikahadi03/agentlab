"""The persistence interface the runtime, tracer, replay, eval harness and dashboard depend on.

Two implementations ship: `SqliteStore` (default, zero setup) and `SupabaseStore` (Postgres over
Supabase's REST API, for a hosted dashboard). Both return identical shapes, which the contract tests enforce:

  run     {run_id, task, provider, model, status, started, ended, meta: dict}
  span    {span_id, run_id, parent_id, kind, name, input, output, status, started, ended, attrs: dict, seq}
  eval    {eval_id, created, suite, meta: dict}
  row     {eval_id, config, task_id, trial, run_id, passed, score, metrics: dict, detail, status, answer}
"""
from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable


class StoreError(RuntimeError):
    pass


@runtime_checkable
class Store(Protocol):
    # runs
    def upsert_run(self, run_id: str, **fields: Any) -> None: ...
    def get_run(self, run_id: str) -> dict | None: ...
    def list_runs(self, limit: int = 20) -> list[dict]: ...
    # spans
    def save_span(self, s: dict) -> None: ...
    def get_spans(self, run_id: str) -> list[dict]: ...
    # checkpoints
    def save_checkpoint(self, run_id: str, seq: int, state: dict) -> None: ...
    def latest_checkpoint(self, run_id: str) -> dict | None: ...
    # evals
    def save_eval_run(self, eval_id: str, suite: str = "", meta: dict | None = None) -> None: ...
    def save_eval_result(
        self, eval_id, config, task_id, trial, run_id, passed, score, metrics, detail, status="", answer=None
    ) -> None: ...
    def list_evals(self, limit: int = 20) -> list[dict]: ...
    def get_eval(self, eval_id: str) -> dict | None: ...
    def get_eval_rows(self, eval_id: str) -> list[dict]: ...
    def close(self) -> None: ...


def make_store(path: str = "agentlab.db", env: dict[str, str] | None = None) -> Store:
    """SqliteStore at `path`, or SupabaseStore when AGENTLAB_STORE=supabase.

    The Supabase store reads SUPABASE_URL and SUPABASE_KEY (a service-role or anon key allowed to use the
    agentlab_* tables). Keys only ever come from the environment, never from files in the repo.
    """
    env = os.environ if env is None else env
    kind = env.get("AGENTLAB_STORE", "sqlite").lower()
    if kind == "sqlite":
        from .store import SqliteStore

        return SqliteStore(path)
    if kind == "supabase":
        from .supabase import SupabaseStore

        url, key = env.get("SUPABASE_URL", ""), env.get("SUPABASE_KEY", "")
        if not url or not key:
            raise StoreError("AGENTLAB_STORE=supabase needs SUPABASE_URL and SUPABASE_KEY in the environment")
        return SupabaseStore(url, key)
    raise StoreError(f"unknown AGENTLAB_STORE {kind!r}; use sqlite or supabase")
