"""A minimal in-process PostgREST look-alike backed by SQLite, for testing SupabaseStore offline.

It implements only what SupabaseStore uses: GET with select/eq/in/order/limit, POST insert, POST upsert
(on_conflict + Prefer: resolution=merge-duplicates), jsonb columns, apikey auth, and injectable failures.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlparse

TABLES = {
    "agentlab_runs": ("run_id TEXT PRIMARY KEY, task TEXT, provider TEXT, model TEXT, status TEXT, "
                      "started REAL, ended REAL, meta TEXT", {"meta"}),
    "agentlab_spans": ("span_id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, kind TEXT, name TEXT, input TEXT, "
                       "output TEXT, status TEXT, started REAL, ended REAL, attrs TEXT, seq INTEGER",
                       {"input", "output", "attrs"}),
    "agentlab_checkpoints": ("run_id TEXT, seq INTEGER, state TEXT, PRIMARY KEY(run_id, seq)", {"state"}),
    "agentlab_eval_runs": ("eval_id TEXT PRIMARY KEY, created REAL, suite TEXT, meta TEXT", {"meta"}),
    "agentlab_eval_results": ("id INTEGER PRIMARY KEY AUTOINCREMENT, eval_id TEXT, config TEXT, task_id TEXT, "
                              "trial INTEGER, run_id TEXT, passed INTEGER, score REAL, metrics TEXT, detail TEXT, "
                              "status TEXT, answer TEXT", {"metrics"}),
}


class FakePostgrest:
    def __init__(self, api_key: str = "test-key"):
        self.api_key = api_key
        self.requests: list[tuple[str, str]] = []  # (method, path+query)
        self.fail_next: list[int] = []  # HTTP statuses to return for the next requests (consumed in order)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        for name, (cols, _) in TABLES.items():
            self._db.execute(f"CREATE TABLE {name}({cols})")
        outer = self

        class Handler(BaseHTTPRequestHandler):
            wbufsize = -1  # send headers + body together: avoids a ~100ms Nagle/delayed-ACK stall per request
            disable_nagle_algorithm = True

            def log_message(self, *a):
                pass

            def _handle(self):
                outer._serve(self)

            do_GET = do_POST = do_PATCH = do_DELETE = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self._thread = threading.Thread(target=lambda: self.httpd.serve_forever(poll_interval=0.01), daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._db.close()

    # request handling ---------------------------------------------------
    def _reply(self, h, code: int, payload=None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _serve(self, h) -> None:
        u = urlparse(h.path)
        length = int(h.headers.get("Content-Length") or 0)
        body = json.loads(h.rfile.read(length)) if length else None
        with self._lock:
            self.requests.append((h.command, h.path))
            if self.fail_next:
                return self._reply(h, self.fail_next.pop(0), {"message": "injected failure"})
            if h.headers.get("apikey") != self.api_key or h.headers.get("Authorization") != f"Bearer {self.api_key}":
                return self._reply(h, 401, {"message": "Invalid API key"})
            prefix = "/rest/v1/"
            table = u.path[len(prefix):] if u.path.startswith(prefix) else ""
            if table not in TABLES:
                return self._reply(h, 404, {"message": f"relation {table} does not exist"})
            params = parse_qsl(u.query, keep_blank_values=True)
            try:
                if h.command == "GET":
                    return self._reply(h, 200, self._select(table, params))
                if h.command == "POST":
                    self._insert(table, params, body, h.headers.get("Prefer", ""))
                    return self._reply(h, 201)
            except sqlite3.Error as e:
                return self._reply(h, 400, {"message": str(e)})
            self._reply(h, 405, {"message": "unsupported"})

    # SQL ----------------------------------------------------------------
    @staticmethod
    def _where(params) -> tuple[str, list]:
        clauses, args = [], []
        for k, v in params:
            if k in ("select", "order", "limit", "on_conflict"):
                continue
            op, _, val = v.partition(".")
            if op == "eq":
                clauses.append(f"{k} = ?")
                args.append(val)
            elif op == "in":
                items = val.strip("()").split(",")
                clauses.append(f"{k} IN ({','.join('?' * len(items))})")
                args += items
            else:
                raise sqlite3.Error(f"unsupported filter {k}={v}")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    def _select(self, table: str, params) -> list[dict]:
        p = dict(params)
        cols = p.get("select", "*")
        where, args = self._where(params)
        sql = f"SELECT {cols} FROM {table}{where}"
        if "order" in p:
            col, _, direction = p["order"].partition(".")
            sql += f" ORDER BY {col} {'DESC' if direction.startswith('desc') else 'ASC'}"
        if "limit" in p:
            sql += f" LIMIT {int(p['limit'])}"
        out = []
        json_cols = TABLES[table][1]
        for row in self._db.execute(sql, args):
            d = dict(row)
            for c in json_cols & d.keys():
                d[c] = json.loads(d[c]) if d[c] is not None else None
            out.append(d)
        return out

    def _insert(self, table: str, params, body, prefer: str) -> None:
        rows = body if isinstance(body, list) else [body]
        json_cols = TABLES[table][1]
        conflict = dict(params).get("on_conflict")
        for row in rows:
            row = {k: (json.dumps(v) if k in json_cols else v) for k, v in row.items()}
            cols = list(row)
            sql = f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
            if conflict and "merge-duplicates" in prefer:
                updates = [c for c in cols if c not in conflict.split(",")]
                sql += f" ON CONFLICT({conflict}) DO " + (
                    "UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in updates) if updates else "NOTHING"
                )
            self._db.execute(sql, list(row.values()))
        self._db.commit()
