"""Dashboard API + server, demo seeding and the serve/demo CLI commands."""
import http.client
import json
import re
import threading

import pytest

import agentlab.demo as demo
from agentlab import AgentRuntime, cli
from agentlab.dashboard import make_server, route
from agentlab.dashboard.server import INDEX, _hostname
from agentlab.eval import EvalConfig, EvalHarness, get_suite
from agentlab.llm import MockAgentProvider
from agentlab.tracing import SqliteStore


@pytest.fixture(scope="module")
def seeded():
    store = SqliteStore(":memory:")
    AgentRuntime(MockAgentProvider(), store=store).run("What is 6*7?", run_id="run-ok")
    AgentRuntime(MockAgentProvider(), store=store).run("<img src=x onerror=alert(1)>", run_id="run-xss")
    h = EvalHarness(store)
    suite = get_suite("injection")
    a = h.run(suite, [EvalConfig("agent", lambda: MockAgentProvider())], suite="injection")
    b = h.run(suite, [EvalConfig("agent", lambda: MockAgentProvider(gullible=True)),
                      EvalConfig("extra", lambda: MockAgentProvider())], suite="injection")
    return store, a.eval_id, b.eval_id


# --- routing ----------------------------------------------------------------
def test_runs_endpoints(seeded):
    store, _, _ = seeded
    code, runs = route(store, "/api/runs", {})
    assert code == 200 and {r["run_id"] for r in runs} >= {"run-ok", "run-xss"}
    assert isinstance(runs[0]["meta"], dict) and runs[0]["meta"]["tokens"] > 0
    code, body = route(store, "/api/runs/run-ok", {})
    assert code == 200 and body["run"]["task"] == "What is 6*7?"
    assert {"run", "plan", "llm", "tool", "critic"} <= {s["kind"] for s in body["spans"]}
    assert route(store, "/api/runs/nope", {})[0] == 404
    assert route(store, "/api/runs", {"limit": ["1"]})[1].__len__() == 1
    assert len(route(store, "/api/runs", {"limit": ["junk"]})[1]) >= 2  # bad limit falls back to the default


def test_eval_endpoints(seeded):
    store, a, b = seeded
    code, evals = route(store, "/api/evals", {})
    assert code == 200 and {e["eval_id"] for e in evals} == {a, b}
    code, d = route(store, f"/api/evals/{b}", {})
    assert code == 200 and d["configs"] == ["agent", "extra"]
    assert d["summary"]["agent"]["success_rate"] == 0.0 and d["summary"]["extra"]["success_rate"] == 1.0
    assert set(d["task_matrix"]) == {"inj-1", "inj-2", "inj-3"}
    assert d["task_metrics"]["inj-1"]["extra"]["pass_rate"] == 1.0
    assert len(d["rows"]) == 6 and "detail" in d["rows"][0] and d["rows"][0]["metrics"]["tokens"] > 0
    assert route(store, "/api/evals/ev-nope", {})[0] == 404


def test_compare_endpoint(seeded):
    store, a, b = seeded
    code, c = route(store, "/api/compare", {"a": [a], "b": [b]})
    c = json.loads(json.dumps(c, default=str))  # what the browser receives
    assert code == 200 and c["configs"]["agent"]["success_rate"] == [1.0, 0.0, -1.0]
    assert c["only_in_b"] == ["extra"] and c["regressions"]
    assert route(store, "/api/compare", {"a": [a]})[0] == 400
    assert route(store, "/api/compare", {"a": [a], "b": ["ev-nope"]})[0] == 404
    assert route(store, "/api/health", {}) == (200, {"ok": True})
    assert route(store, "/api/unknown", {})[0] == 404


def test_hostname_parsing():
    assert _hostname("127.0.0.1:8765") == "127.0.0.1" and _hostname("localhost") == "localhost"
    assert _hostname("[::1]:8765") == "[::1]" and _hostname("evil.example:80") == "evil.example"


# --- HTTP server ------------------------------------------------------------
@pytest.fixture
def server(seeded):
    srv = make_server(seeded[0], "127.0.0.1", 0)
    t = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
    t.start()
    yield srv.server_port
    srv.shutdown()
    srv.server_close()


def _get(port, path, method="GET", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, r, body


def test_server_serves_page_and_api(server):
    status, r, body = _get(server, "/")
    assert status == 200 and r.getheader("Content-Type").startswith("text/html") and b"<title>AgentLab</title>" in body
    status, r, body = _get(server, "/api/runs")
    assert status == 200 and r.getheader("Content-Type") == "application/json"
    # stored verbatim; the page escapes it, and nosniff + JSON content type stop it being treated as HTML
    assert any(x["task"] == "<img src=x onerror=alert(1)>" for x in json.loads(body))
    assert _get(server, "/api/runs/run-ok")[0] == 200
    assert _get(server, "/missing")[0] == 404


def test_server_security_headers_and_read_only(server):
    _, r, _ = _get(server, "/")
    assert r.getheader("X-Content-Type-Options") == "nosniff"
    assert "default-src 'self'" in r.getheader("Content-Security-Policy")
    assert r.getheader("Cache-Control") == "no-store"
    for m in ("POST", "PUT", "DELETE", "PATCH"):
        assert _get(server, "/api/runs", method=m)[0] == 405


def test_server_rejects_foreign_host_header(server):
    assert _get(server, "/api/runs", headers={"Host": "evil.example"})[0] == 403  # DNS-rebinding guard
    assert _get(server, "/api/runs", headers={"Host": f"localhost:{server}"})[0] == 200


def test_server_survives_store_errors(seeded):
    class Broken:
        def list_runs(self, limit=20):
            raise RuntimeError("db down")

    srv = make_server(Broken(), "127.0.0.1", 0)
    threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True).start()
    try:
        status, _, body = _get(srv.server_port, "/api/runs")
        assert status == 500 and "db down" in json.loads(body)["error"]
        assert _get(srv.server_port, "/api/health")[0] == 200  # still alive
    finally:
        srv.shutdown()
        srv.server_close()


def test_page_is_self_contained_and_escapes_content():
    html = INDEX.read_text(encoding="utf-8")
    assert not re.search(r"""(?:src|href)=["']https?://""", html)  # no CDN or tracker
    assert "esc(" in html and "innerHTML" in html
    # every dynamic string reaching innerHTML in the run view goes through esc(): spot-check the riskiest fields
    for field in ("run.task", "s.name", "r.task", "r.detail"):
        assert re.search(rf"esc\({re.escape(field)}\)", html), field
    assert 'data-theme="dark"' in html and "prefers-color-scheme: dark" in html  # both theme paths exist


# --- demo + cli -------------------------------------------------------------
def test_seed_demo_creates_runs_and_comparable_evals(monkeypatch):
    monkeypatch.setattr(demo, "get_suite", lambda name: get_suite("injection") + get_suite("error-recovery"))
    store = SqliteStore(":memory:")
    assert demo.is_empty(store)
    msgs = []
    out = demo.seed_demo(store, progress=msgs.append)
    assert not demo.is_empty(store) and msgs
    statuses = {store.get_run(r)["status"] for r in out["runs"]}
    assert {"success", "failed", "budget_exceeded"} <= statuses
    assert store.get_run(out["runs"][0])["meta"]["cost_usd"] > 0  # priced mock => visible cost
    code, c = route(store, "/api/compare", {"a": [out["baseline"]], "b": [out["candidate"]]})
    c = json.loads(json.dumps(c, default=str))
    assert code == 200 and c["configs"]["agent"]["success_rate"][2] < 0  # the planted regression
    assert c["configs"]["agent-flaky"]["success_rate"][2] == 0 and c["only_in_b"] == ["agent-brittle"]


def test_cli_serve_and_demo(tmp_path, monkeypatch, capsys):
    calls = []
    opened = []
    monkeypatch.setattr("agentlab.dashboard.serve", lambda store, host, port, on_ready=None: (
        calls.append((host, port)), on_ready and on_ready(f"http://{host}:{port}")))
    monkeypatch.setattr("webbrowser.open", opened.append)
    db = str(tmp_path / "t.db")
    assert cli.main(["--db", db, "serve", "--port", "9999"]) == 0
    assert calls == [("127.0.0.1", 9999)] and opened == ["http://127.0.0.1:9999"]
    assert cli.main(["--db", db, "serve", "--no-open", "--host", "0.0.0.0"]) == 0
    assert opened == ["http://127.0.0.1:9999"]  # --no-open respected

    seeded_into = []
    monkeypatch.setattr(demo, "seed_demo", lambda store, progress=None, workers=4: seeded_into.append(store))
    demo_db = str(tmp_path / "demo.db")
    assert cli.main(["demo", "--demo-db", demo_db, "--no-open"]) == 0
    assert len(seeded_into) == 1 and "Seeding demo data" in capsys.readouterr().err
    seeded_into[0].upsert_run("x", task="t")  # pretend seeding happened; a second launch must not reseed
    seeded_into[0].close()
    assert cli.main(["demo", "--demo-db", demo_db, "--no-open"]) == 0
    assert len(seeded_into) == 1 and "already present" in capsys.readouterr().err


def test_demo_seed_only_exits_and_port_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "seed_demo", lambda store, progress=None, workers=4: store.upsert_run("x", task="t"))
    monkeypatch.setattr("agentlab.dashboard.serve", lambda *a, **k: pytest.fail("must not serve with --seed-only"))
    assert cli.main(["demo", "--demo-db", str(tmp_path / "d.db"), "--seed-only"]) == 0
    assert SqliteStore(str(tmp_path / "d.db")).get_run("x")["task"] == "t"
    monkeypatch.setenv("PORT", "7860")
    assert cli._default_port() == 7860
    monkeypatch.setenv("PORT", "not-a-number")
    assert cli._default_port() == 8765
    monkeypatch.delenv("PORT")
    assert cli._default_port() == 8765
