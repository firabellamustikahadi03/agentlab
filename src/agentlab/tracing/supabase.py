"""SupabaseStore: the Store interface over Supabase's PostgREST API (Postgres), stdlib only.

Create the tables once with docs/supabase_schema.sql, then:
    export AGENTLAB_STORE=supabase SUPABASE_URL=https://<ref>.supabase.co SUPABASE_KEY=<key>
Every Store method is one REST call (listing evals takes two). Spans are written once, when they finish.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .base import StoreError

_RETRY = (429, 500, 502, 503, 504)


class SupabaseStore:
    def __init__(
        self,
        url: str,
        key: str,
        prefix: str = "agentlab_",
        timeout: float = 30.0,
        max_retries: int = 4,
    ):
        self.base = url.rstrip("/") + "/rest/v1/"
        self.key, self.prefix = key, prefix
        self.timeout, self.max_retries = timeout, max_retries

    # transport ----------------------------------------------------------
    def _request(
        self, method: str, table: str, params: dict[str, str] | None = None, body: Any = None, prefer: str = ""
    ) -> list[dict]:
        url = self.base + self.prefix + table
        if params:
            url += "?" + urllib.parse.urlencode(params, safe="(),.*")
        headers = {
            "apikey": self.key, "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        data = json.dumps(body, default=str).encode() if body is not None else None
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                return json.loads(raw) if raw else []
            except urllib.error.HTTPError as e:
                last = e
                if e.code in _RETRY:
                    time.sleep(min(2**attempt, 10))
                    continue
                raise StoreError(f"supabase {method} {table}: HTTP {e.code} {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(min(2**attempt, 10))
        raise StoreError(f"supabase {method} {table} failed after {self.max_retries} attempts: {last}")

    def _upsert(self, table: str, row: dict, on_conflict: str) -> None:
        self._request(
            "POST", table, {"on_conflict": on_conflict}, row, prefer="resolution=merge-duplicates,return=minimal"
        )

    def _one(self, table: str, **eq: str) -> dict | None:
        rows = self._request("GET", table, {"select": "*", "limit": "1", **{k: f"eq.{v}" for k, v in eq.items()}})
        return rows[0] if rows else None

    # runs ---------------------------------------------------------------
    def upsert_run(self, run_id: str, **f: Any) -> None:
        self._upsert("runs", {"run_id": run_id, **f}, "run_id")

    def get_run(self, run_id: str) -> dict | None:
        r = self._one("runs", run_id=run_id)
        return self._norm_run(r) if r else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = self._request("GET", "runs", {"select": "*", "order": "started.desc.nullslast", "limit": str(limit)})
        return [self._norm_run(r) for r in rows]

    @staticmethod
    def _norm_run(r: dict) -> dict:
        r["meta"] = r.get("meta") or {}
        return r

    # spans --------------------------------------------------------------
    def save_span(self, s: dict) -> None:
        self._upsert("spans", s, "span_id")

    def get_spans(self, run_id: str) -> list[dict]:
        return self._request("GET", "spans", {"select": "*", "run_id": f"eq.{run_id}", "order": "seq.asc"})

    # checkpoints --------------------------------------------------------
    def save_checkpoint(self, run_id: str, seq: int, state: dict) -> None:
        self._upsert("checkpoints", {"run_id": run_id, "seq": seq, "state": state}, "run_id,seq")

    def latest_checkpoint(self, run_id: str) -> dict | None:
        rows = self._request(
            "GET", "checkpoints", {"select": "state", "run_id": f"eq.{run_id}", "order": "seq.desc", "limit": "1"}
        )
        return rows[0]["state"] if rows else None

    # eval ---------------------------------------------------------------
    def save_eval_run(self, eval_id: str, suite: str = "", meta: dict | None = None) -> None:
        self._upsert(
            "eval_runs", {"eval_id": eval_id, "created": time.time(), "suite": suite, "meta": meta or {}}, "eval_id"
        )

    def save_eval_result(
        self, eval_id, config, task_id, trial, run_id, passed, score, metrics, detail, status="", answer=None
    ) -> None:
        row = {
            "eval_id": eval_id, "config": config, "task_id": task_id, "trial": trial, "run_id": run_id,
            "passed": int(passed), "score": score, "metrics": metrics, "detail": detail, "status": status,
            "answer": answer,
        }
        self._request("POST", "eval_results", body=row, prefer="return=minimal")

    def list_evals(self, limit: int = 20) -> list[dict]:
        evals = self._request("GET", "eval_runs", {"select": "*", "order": "created.desc", "limit": str(limit)})
        if not evals:
            return []
        ids = ",".join(e["eval_id"] for e in evals)
        results = self._request("GET", "eval_results", {"select": "eval_id,passed", "eval_id": f"in.({ids})"})
        counts: dict[str, list[int]] = {}
        for r in results:
            counts.setdefault(r["eval_id"], []).append(r["passed"])
        for e in evals:
            e["meta"] = e.get("meta") or {}
            v = counts.get(e["eval_id"], [])
            e["n"] = len(v)
            e["success"] = (sum(v) / len(v)) if v else None
        return evals

    def get_eval(self, eval_id: str) -> dict | None:
        e = self._one("eval_runs", eval_id=eval_id)
        if e:
            e["meta"] = e.get("meta") or {}
        return e

    def get_eval_rows(self, eval_id: str) -> list[dict]:
        rows = self._request("GET", "eval_results", {"select": "*", "eval_id": f"eq.{eval_id}", "order": "id.asc"})
        for r in rows:
            r.pop("id", None)
            r["status"] = r.get("status") or ""
        return rows

    def close(self) -> None:
        pass
