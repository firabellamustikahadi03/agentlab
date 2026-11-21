"""SQLite persistence for runs, spans and checkpoints (swap for Supabase/Postgres later)."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, task TEXT, provider TEXT, model TEXT, status TEXT,
  started REAL, ended REAL, meta TEXT
);
CREATE TABLE IF NOT EXISTS spans(
  span_id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, kind TEXT, name TEXT,
  input TEXT, output TEXT, status TEXT, started REAL, ended REAL, attrs TEXT, seq INTEGER
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id, seq);
CREATE TABLE IF NOT EXISTS checkpoints(
  run_id TEXT, seq INTEGER, state TEXT, PRIMARY KEY(run_id, seq)
);
CREATE TABLE IF NOT EXISTS eval_results(
  eval_id TEXT, config TEXT, task_id TEXT, trial INTEGER, run_id TEXT, passed INTEGER,
  score REAL, metrics TEXT, detail TEXT
);
"""


class SqliteStore:
    def __init__(self, path: str = "agentlab.db"):
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur.fetchall()

    # runs ---------------------------------------------------------------
    def upsert_run(self, run_id: str, **f: Any) -> None:
        if not self._exec("SELECT 1 FROM runs WHERE run_id=?", (run_id,)):
            self._exec("INSERT INTO runs(run_id) VALUES(?)", (run_id,))
        for k, v in f.items():
            if k == "meta":
                v = json.dumps(v, default=str)
            self._exec(f"UPDATE runs SET {k}=? WHERE run_id=?", (v, run_id))

    def get_run(self, run_id: str) -> dict | None:
        r = self._exec("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return dict(r[0]) if r else None

    def list_runs(self, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self._exec("SELECT * FROM runs ORDER BY started DESC LIMIT ?", (limit,))]

    # spans --------------------------------------------------------------
    def save_span(self, s: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO spans VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                s["span_id"], s["run_id"], s["parent_id"], s["kind"], s["name"],
                json.dumps(s["input"], default=str), json.dumps(s["output"], default=str),
                s["status"], s["started"], s["ended"], json.dumps(s["attrs"], default=str), s["seq"],
            ),
        )

    def get_spans(self, run_id: str) -> list[dict]:
        out = []
        for r in self._exec("SELECT * FROM spans WHERE run_id=? ORDER BY seq", (run_id,)):
            d = dict(r)
            for k in ("input", "output", "attrs"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    # checkpoints --------------------------------------------------------
    def save_checkpoint(self, run_id: str, seq: int, state: dict) -> None:
        self._exec("INSERT OR REPLACE INTO checkpoints VALUES(?,?,?)", (run_id, seq, json.dumps(state, default=str)))

    def latest_checkpoint(self, run_id: str) -> dict | None:
        r = self._exec("SELECT state FROM checkpoints WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run_id,))
        return json.loads(r[0]["state"]) if r else None

    # eval ---------------------------------------------------------------
    def save_eval_result(self, eval_id, config, task_id, trial, run_id, passed, score, metrics, detail) -> None:
        self._exec(
            "INSERT INTO eval_results VALUES(?,?,?,?,?,?,?,?,?)",
            (eval_id, config, task_id, trial, run_id, int(passed), score, json.dumps(metrics), detail),
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()
