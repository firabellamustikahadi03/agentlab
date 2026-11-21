"""Read-only web dashboard: a tiny stdlib HTTP server (JSON API + one static page, no build step).

Security posture: GET only; binds to 127.0.0.1 by default; when bound to loopback the Host header must be a
loopback name (blocks DNS rebinding); the page escapes all trace content (it comes from models and tools).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..eval import EvalReport, compare_reports
from ..tracing.base import Store

INDEX = Path(__file__).with_name("index.html")
_LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
_CSP = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:"


def eval_payload(store: Store, eval_id: str) -> dict | None:
    meta = store.get_eval(eval_id)
    if meta is None:
        return None
    rep = EvalReport.from_store(store, eval_id)
    rows = [{k: r[k] for k in ("config", "task_id", "trial", "run_id", "status", "passed", "score", "detail", "answer")}
            | {"metrics": r["metrics"]} for r in rep.rows]
    return {
        "eval": meta, "configs": rep.configs(), "summary": rep.summary(), "task_matrix": rep.task_matrix(),
        "task_metrics": rep.task_metrics(), "rows": rows,
    }


def route(store: Store, path: str, query: dict[str, list[str]]) -> tuple[int, dict | list]:
    """Pure routing: (status, JSON-able payload). Kept separate from HTTP so it is trivially testable."""
    if path == "/api/health":
        return 200, {"ok": True}
    if path == "/api/runs":
        limit = _int(query.get("limit", ["200"])[0], 200, 1, 1000)
        return 200, store.list_runs(limit)
    if m := re.fullmatch(r"/api/runs/([\w.-]+)", path):
        run = store.get_run(m.group(1))
        return (200, {"run": run, "spans": store.get_spans(m.group(1))}) if run else (404, {"error": "run not found"})
    if path == "/api/evals":
        return 200, store.list_evals(_int(query.get("limit", ["100"])[0], 100, 1, 500))
    if m := re.fullmatch(r"/api/evals/([\w.-]+)", path):
        payload = eval_payload(store, m.group(1))
        return (200, payload) if payload else (404, {"error": "eval not found"})
    if path == "/api/compare":
        a, b = query.get("a", [""])[0], query.get("b", [""])[0]
        if not a or not b:
            return 400, {"error": "need ?a=<eval_id>&b=<eval_id>"}
        if store.get_eval(a) is None or store.get_eval(b) is None:
            return 404, {"error": "eval not found"}
        cmp = compare_reports(EvalReport.from_store(store, a), EvalReport.from_store(store, b))
        return 200, asdict(cmp) | {"regressions": cmp.regressions()}
    return 404, {"error": "not found"}


def _hostname(header: str) -> str:
    if header.startswith("["):  # IPv6 literal, e.g. [::1]:8765
        return header[: header.find("]") + 1]
    return header.rsplit(":", 1)[0] if ":" in header else header


def _int(v: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except ValueError:
        return default


def make_handler(store: Store, loopback_only: bool) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        wbufsize = -1  # headers + body in one packet (avoids a Nagle/delayed-ACK stall on every request)
        disable_nagle_algorithm = True
        server_version = "agentlab-dashboard"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", _CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            if loopback_only and _hostname(self.headers.get("Host") or "") not in _LOOPBACK:
                return self._json(403, {"error": "forbidden host"})
            u = urlparse(self.path)
            path = unquote(u.path)
            if path in ("/", "/index.html"):
                return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/api/"):
                try:
                    code, payload = route(store, path, parse_qs(u.query))
                except Exception as e:  # a broken store must not kill the server
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return self._json(code, payload)
            self._json(404, {"error": "not found"})

        def _no(self):
            self._json(405, {"error": "read-only dashboard"})

        do_POST = do_PUT = do_PATCH = do_DELETE = _no  # noqa: N815

    return Handler


def make_server(store: Store, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    loopback = host in ("127.0.0.1", "localhost", "::1")
    srv = ThreadingHTTPServer((host, port), make_handler(store, loopback))
    srv.daemon_threads = True
    return srv


def serve(store: Store, host: str = "127.0.0.1", port: int = 8765, on_ready=None) -> None:
    srv = make_server(store, host, port)
    if on_ready:
        on_ready(f"http://{host}:{srv.server_port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
