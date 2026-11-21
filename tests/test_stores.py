"""Store contract tests: every behaviour is checked against SqliteStore AND SupabaseStore (via a fake PostgREST)."""
import pytest

from agentlab import AgentRuntime, cli
from agentlab.eval import EvalConfig, EvalHarness, EvalReport, compare_reports, get_suite
from agentlab.llm import MockAgentProvider
from agentlab.tracing import ReplayProvider, SqliteStore, Store, StoreError, SupabaseStore, make_store
from fake_postgrest import FakePostgrest


@pytest.fixture
def fake():
    f = FakePostgrest()
    yield f
    f.close()


@pytest.fixture(params=["sqlite", "supabase"])
def store(request, fake):
    s = SqliteStore(":memory:") if request.param == "sqlite" else SupabaseStore(fake.url, fake.api_key)
    yield s
    s.close()


def _span(i, run="r1", parent=None, **kw):
    d = {"span_id": f"s{i}", "run_id": run, "parent_id": parent, "kind": "llm", "name": f"n{i}",
         "input": {"q": i}, "output": {"a": [i, "x"]}, "status": "ok", "started": 100.0 + i, "ended": 101.0 + i,
         "attrs": {"tokens": i}, "seq": i}
    d.update(kw)
    return d


# --- contract ---------------------------------------------------------------
def test_satisfies_protocol(store):
    assert isinstance(store, Store)


def test_runs_merge_fields_and_list_newest_first(store):
    store.upsert_run("a", task="t1", provider="p", model="m", status="running", started=1.0)
    store.upsert_run("a", status="success", ended=2.0, meta={"tokens": 5})  # partial update keeps other fields
    store.upsert_run("b", task="t2", started=5.0)
    store.upsert_run("c", task="t3")  # no start time: sorted last
    a = store.get_run("a")
    assert (a["task"], a["status"], a["started"], a["ended"], a["meta"]) == ("t1", "success", 1.0, 2.0, {"tokens": 5})
    assert store.get_run("b")["meta"] == {}
    assert store.get_run("nope") is None
    assert [r["run_id"] for r in store.list_runs()] == ["b", "a", "c"]
    assert [r["run_id"] for r in store.list_runs(limit=1)] == ["b"]


def test_spans_roundtrip_ordered_by_seq_and_upsert(store):
    for i in (2, 0, 1):
        store.save_span(_span(i, parent=None if i == 0 else "s0"))
    store.save_span(_span(1, parent="s0", status="error", output=None))  # same span_id: replaced, not duplicated
    store.save_span(_span(9, run="other"))
    spans = store.get_spans("r1")
    assert [s["seq"] for s in spans] == [0, 1, 2]
    assert spans[1]["status"] == "error" and spans[1]["output"] is None
    assert (spans[2]["input"], spans[2]["output"], spans[2]["attrs"]) == ({"q": 2}, {"a": [2, "x"]}, {"tokens": 2})
    assert spans[0]["parent_id"] is None and spans[1]["parent_id"] == "s0"
    assert store.get_spans("missing") == []


def test_checkpoints_latest_and_overwrite(store):
    assert store.latest_checkpoint("r") is None
    store.save_checkpoint("r", 1, {"step": 1})
    store.save_checkpoint("r", 3, {"step": 3, "nested": {"x": [1, 2]}})
    store.save_checkpoint("r", 2, {"step": 2})
    store.save_checkpoint("r", 3, {"step": 3, "nested": "replaced"})
    assert store.latest_checkpoint("r") == {"step": 3, "nested": "replaced"}
    assert store.latest_checkpoint("other") is None


def test_eval_runs_results_and_listing(store):
    store.save_eval_run("ev-1", "basic", {"configs": ["m"]})
    store.save_eval_run("ev-2", "all", {"configs": ["m", "n"]})
    for i, passed in enumerate([True, False, True]):
        store.save_eval_result("ev-1", "m", f"t{i}", 0, f"run{i}", passed, float(passed), {"tokens": i}, f"d{i}",
                               status="success", answer=str(i))
    store.save_eval_result("ev-1", "m", "t9", 1, "run9", True, 1.0, {}, "", status="", answer=None)
    rows = store.get_eval_rows("ev-1")
    assert [r["task_id"] for r in rows] == ["t0", "t1", "t2", "t9"]  # insertion order
    assert rows[0] == {"eval_id": "ev-1", "config": "m", "task_id": "t0", "trial": 0, "run_id": "run0", "passed": 1,
                       "score": 1.0, "metrics": {"tokens": 0}, "detail": "d0", "status": "success", "answer": "0"}
    assert rows[3]["answer"] is None and rows[3]["status"] == ""
    assert store.get_eval_rows("ev-2") == []
    e = store.get_eval("ev-1")
    assert (e["suite"], e["meta"]) == ("basic", {"configs": ["m"]}) and e["created"] > 0
    assert store.get_eval("nope") is None
    listing = {e["eval_id"]: e for e in store.list_evals()}
    assert listing["ev-1"]["n"] == 4 and listing["ev-1"]["success"] == 1.0 * 3 / 4
    assert listing["ev-2"]["n"] == 0 and listing["ev-2"]["success"] is None
    assert listing["ev-1"]["meta"] == {"configs": ["m"]}
    assert len(store.list_evals(limit=1)) == 1
    assert store.list_evals(limit=1)[0]["eval_id"] == "ev-2"  # newest first


def test_empty_listing(store):
    assert store.list_runs() == [] and store.list_evals() == []


# --- the whole product on each store ----------------------------------------
class _CrashAfter:
    name, model = "crash", "crash-1"

    def __init__(self, inner, n):
        self.inner, self.n, self.i = inner, n, 0

    def complete(self, messages, temperature=0.0):
        self.i += 1
        if self.i == self.n:
            raise ConnectionError("boom")
        return self.inner.complete(messages, temperature)


def test_runtime_trace_resume_and_replay_on_any_store(store):
    res = AgentRuntime(MockAgentProvider(), store=store).run("What is 17*23+5?")
    assert res.ok and res.answer == "396"
    assert {"run", "plan", "step", "llm", "tool", "critic"} <= {s["kind"] for s in store.get_spans(res.run_id)}
    assert store.get_run(res.run_id)["status"] == "success"

    crashed = AgentRuntime(_CrashAfter(MockAgentProvider(), 4), store=store).run("What is 12*12?")
    assert crashed.status == "failed" and store.get_run(crashed.run_id)["status"] == "failed"
    resumed = AgentRuntime(MockAgentProvider(), store=store).resume(crashed.run_id)
    assert resumed.ok and resumed.answer == "144" and resumed.metrics["llm_calls"] == 4

    rp = ReplayProvider(store, res.run_id)
    again = AgentRuntime(rp, store=store).run("What is 17*23+5?")
    assert again.answer == "396" and rp.divergences == []


def test_eval_persist_reload_compare_on_any_store(store):
    h = EvalHarness(store)
    good = h.run(get_suite("injection"), [EvalConfig("agent", lambda: MockAgentProvider())], suite="injection")
    bad = h.run(get_suite("injection"), [EvalConfig("agent", lambda: MockAgentProvider(gullible=True))], workers=3)
    loaded = EvalReport.from_store(store, good.eval_id)
    assert loaded.summary() == good.summary()
    cmp = compare_reports(loaded, EvalReport.from_store(store, bad.eval_id))
    assert cmp.configs["agent"]["success_rate"][2] == -1.0 and cmp.regressions()


def test_sqlite_and_supabase_agree_on_identical_work(fake):
    sq, sb = SqliteStore(":memory:"), SupabaseStore(fake.url, fake.api_key)
    out = []
    for s in (sq, sb):
        res = AgentRuntime(MockAgentProvider(fail_first=True), store=s).run("What is 7*8*9?", run_id="same")
        spans = [(x["kind"], x["name"], x["status"], x["seq"], x["parent_id"] is None) for x in s.get_spans("same")]
        out.append((res.status, res.answer, res.metrics["llm_calls"], res.metrics["replans"], spans))
    assert out[0] == out[1]


# --- supabase transport -----------------------------------------------------
def test_requests_carry_auth_and_use_upserts(fake):
    s = SupabaseStore(fake.url, fake.api_key)
    s.upsert_run("r", task="t")
    method, path = fake.requests[-1]
    assert method == "POST" and "/rest/v1/agentlab_runs" in path and "on_conflict=run_id" in path


def test_bad_key_is_a_clear_non_retried_error(fake):
    s = SupabaseStore(fake.url, "wrong-key")
    with pytest.raises(StoreError, match="HTTP 401"):
        s.list_runs()
    assert len(fake.requests) == 1  # auth errors are not retried


def test_retries_transient_errors_then_succeeds(fake, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = SupabaseStore(fake.url, fake.api_key)
    fake.fail_next = [503, 429]
    s.upsert_run("r", task="t")
    assert s.get_run("r")["task"] == "t"
    assert len([r for r in fake.requests if r[0] == "POST"]) == 3  # 2 failures + 1 success


def test_gives_up_after_max_retries(fake, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = SupabaseStore(fake.url, fake.api_key, max_retries=2)
    fake.fail_next = [500, 500, 500]
    with pytest.raises(StoreError, match="after 2 attempts"):
        s.list_runs()


def test_unreachable_server_raises_store_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = SupabaseStore("http://127.0.0.1:9", "k", max_retries=2, timeout=0.5)
    with pytest.raises(StoreError):
        s.list_runs()


def test_unknown_table_is_reported(fake):
    s = SupabaseStore(fake.url, fake.api_key, prefix="missing_")
    with pytest.raises(StoreError, match="HTTP 404"):
        s.list_runs()


# --- factory + CLI ----------------------------------------------------------
def test_make_store_selection(tmp_path, fake):
    assert isinstance(make_store(str(tmp_path / "x.db"), env={}), SqliteStore)
    assert isinstance(make_store(env={"AGENTLAB_STORE": "supabase", "SUPABASE_URL": fake.url, "SUPABASE_KEY": "k"}),
                      SupabaseStore)
    with pytest.raises(StoreError, match="SUPABASE_URL"):
        make_store(env={"AGENTLAB_STORE": "supabase"})
    with pytest.raises(StoreError, match="unknown AGENTLAB_STORE"):
        make_store(env={"AGENTLAB_STORE": "mongo"})


def test_cli_runs_against_supabase_via_env(fake, monkeypatch, capsys):
    monkeypatch.setenv("AGENTLAB_STORE", "supabase")
    monkeypatch.setenv("SUPABASE_URL", fake.url)
    monkeypatch.setenv("SUPABASE_KEY", fake.api_key)
    assert cli.main(["run", "What is 6*7?"]) == 0
    capsys.readouterr()
    assert cli.main(["traces", "list"]) == 0
    assert "What is 6*7?" in capsys.readouterr().out
    assert cli.main(["eval", "--suite", "injection", "--providers", "mock"]) == 0
    capsys.readouterr()
    assert cli.main(["eval-list"]) == 0
    assert "injection" in capsys.readouterr().out
