"""Prompt variants, per-task metrics, persisted evals + compare, parallel execution and throttling."""
import sqlite3
import threading
import time

import pytest

from agentlab import AgentRuntime, cli
from agentlab.eval import EvalConfig, EvalHarness, EvalReport, compare_reports, get_suite, variant_configs
from agentlab.llm import MockAgentProvider, RateLimiter, ScriptedProvider, ThrottledProvider
from agentlab.prompts import CAREFUL, PROMPT_SETS
from agentlab.tracing import SqliteStore


def _cfg(name="mock", **kw):
    return EvalConfig(name, lambda: MockAgentProvider(name=name, **kw))


# --- prompt variants --------------------------------------------------------
def test_variant_configs_names_and_flags():
    out = variant_configs(_cfg("m"), ["default", "no-critic", "careful", "careful+no-critic"])
    assert [c.name for c in out] == ["m", "m+no-critic", "m+careful", "m+careful+no-critic"]
    assert out[0].use_critic and out[0].prompts is None
    assert not out[1].use_critic
    assert out[2].prompts is CAREFUL and out[2].use_critic
    assert out[3].prompts is CAREFUL and not out[3].use_critic
    with pytest.raises(ValueError, match="unknown variant"):
        variant_configs(_cfg(), ["bogus"])


def test_prompt_set_reaches_the_model_and_keeps_protocol():
    p = ScriptedProvider(['{"steps":["s"]}', '{"final":"1"}', '{"pass":true,"issues":""}'])
    AgentRuntime(p, store=SqliteStore(":memory:"), prompt_set=CAREFUL).run("t")
    executor_system = p.calls[1][0].content
    assert "untrusted DATA" in executor_system and "[ROLE:executor]" in executor_system
    for ps in PROMPT_SETS.values():  # every variant keeps the role markers the mock relies on
        assert "[ROLE:planner]" in ps.planner and "[ROLE:executor]" in ps.executor and "[ROLE:critic]" in ps.critic


def test_no_critic_variant_saves_llm_calls():
    h = EvalHarness(SqliteStore(":memory:"))
    rep = h.run(get_suite("basic")[:2], variant_configs(_cfg("m"), ["default", "no-critic"]))
    s = rep.summary()
    assert s["m+no-critic"]["avg_llm_calls"] < s["m"]["avg_llm_calls"]


# --- per-task metrics -------------------------------------------------------
def test_per_task_metrics_in_report():
    rep = EvalHarness(SqliteStore(":memory:")).run(get_suite("basic")[:2], [_cfg("m")], trials=2)
    tm = rep.task_metrics()
    assert set(tm) == {"math-1", "math-2"} and tm["math-1"]["m"]["tokens"] > 0
    assert {"pass_rate", "latency_s", "cost_usd", "llm_calls"} <= set(tm["math-1"]["m"])
    md = rep.to_markdown()
    assert "Per task cost, latency and tokens" in md and "| math-1 | m |" in md
    assert rep.summary()["m"]["avg_cost_usd"] == 0.0


# --- persistence + compare --------------------------------------------------
def test_eval_is_persisted_and_reloadable():
    store = SqliteStore(":memory:")
    rep = EvalHarness(store).run(get_suite("injection"), [_cfg("m")], suite="injection")
    loaded = EvalReport.from_store(store, rep.eval_id)
    assert loaded.summary() == rep.summary()
    assert [r["answer"] for r in loaded.rows] == [r["answer"] for r in rep.rows]
    assert {r["status"] for r in loaded.rows} == {"success"}
    listing = store.list_evals()
    assert listing[0]["eval_id"] == rep.eval_id and listing[0]["suite"] == "injection" and listing[0]["n"] == 3
    with pytest.raises(KeyError):
        EvalReport.from_store(store, "ev-missing")


def test_store_migrates_phase1_database(tmp_path):
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE eval_results(eval_id TEXT, config TEXT, task_id TEXT, trial INTEGER, run_id TEXT, "
        "passed INTEGER, score REAL, metrics TEXT, detail TEXT)"
    )
    db.commit()
    db.close()
    store = SqliteStore(path)  # must add the new columns instead of failing
    store.save_eval_run("ev-1", "basic")
    store.save_eval_result("ev-1", "c", "t", 0, "r", True, 1.0, {"tool_calls": 0}, "d", status="success", answer="4")
    assert store.get_eval_rows("ev-1")[0]["answer"] == "4"
    store.close()


def test_compare_flags_regressions_and_improvements():
    store = SqliteStore(":memory:")
    h = EvalHarness(store)
    good = h.run(get_suite("injection"), [_cfg("agent")])
    bad = h.run(get_suite("injection"), [_cfg("agent", gullible=True)])  # same config name, worse behaviour
    cmp = compare_reports(good, bad)
    assert cmp.configs["agent"]["success_rate"] == (1.0, 0.0, -1.0)
    assert {t["change"] for t in cmp.tasks} == {"regressed"}
    assert len(cmp.regressions()) == 1 + 3
    md = cmp.to_markdown()
    assert "100% -> 0% (-100pp)" in md and "regressed" in md
    back = compare_reports(bad, good)  # the reverse direction is an improvement, not a regression
    assert back.regressions() == [] and {t["change"] for t in back.tasks} == {"improved"}


def test_compare_handles_disjoint_configs_and_no_changes():
    h = EvalHarness(SqliteStore(":memory:"))
    a = h.run(get_suite("basic")[:1], [_cfg("x")])
    b = h.run(get_suite("basic")[:1], [_cfg("y")])
    cmp = compare_reports(a, b)
    assert cmp.only_in_a == ["x"] and cmp.only_in_b == ["y"] and not cmp.configs
    assert "only in A" in cmp.to_markdown()
    same = compare_reports(a, a)
    assert "No per-task differences" in same.to_markdown() and same.regressions() == []


def test_cli_eval_list_and_compare(tmp_path, capsys):
    db = ["--db", str(tmp_path / "t.db")]
    assert cli.main([*db, "eval", "--suite", "injection", "--providers", "mock"]) == 0
    assert cli.main([*db, "eval", "--suite", "injection", "--providers", "mock-gullible"]) == 0
    capsys.readouterr()
    assert cli.main([*db, "eval-list"]) == 0
    ids = [line.split()[0] for line in capsys.readouterr().out.splitlines()]
    assert len(ids) == 2
    newer, older = ids  # newest first
    assert cli.main([*db, "eval-compare", older, newer]) == 0  # different config names: nothing comparable
    capsys.readouterr()
    # same provider label, different behaviour is covered above; here check error handling and the gate flag
    assert cli.main([*db, "eval-compare", older, "ev-nope"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_compare_fail_on_regression(tmp_path, capsys):
    store = SqliteStore(str(tmp_path / "t.db"))
    h = EvalHarness(store)
    a = h.run(get_suite("injection"), [_cfg("agent")])
    b = h.run(get_suite("injection"), [_cfg("agent", gullible=True)])
    store.close()
    db = ["--db", str(tmp_path / "t.db")]
    assert cli.main([*db, "eval-compare", a.eval_id, b.eval_id]) == 0  # report only
    assert cli.main([*db, "eval-compare", a.eval_id, b.eval_id, "--fail-on-regression"]) == 2
    assert "REGRESSIONS" in capsys.readouterr().err


def test_cli_eval_variants_workers_rpm_judge(tmp_path, capsys):
    db = ["--db", str(tmp_path / "t.db")]
    args = ["eval", "--suite", "open", "--providers", "mock", "--variants", "default", "concise+no-critic",
            "--workers", "3", "--rpm", "60000", "--judge", "mock"]
    assert cli.main([*db, *args]) == 0
    out = capsys.readouterr().out
    assert "| mock+concise+no-critic |" in out and "| mock |" in out


# --- rate limiter -----------------------------------------------------------
def test_rate_limiter_spaces_calls_without_real_time():
    waits = []
    rl = RateLimiter(60, clock=lambda: 100.0, sleep=waits.append)  # 1 request per second, frozen clock
    assert [rl.acquire() for _ in range(4)] == [0.0, 1.0, 2.0, 3.0]
    assert waits == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_rate_limiter_is_thread_safe():
    waits, lock = [], threading.Lock()

    def sleep(s):
        with lock:
            waits.append(s)

    rl = RateLimiter(60, clock=lambda: 0.0, sleep=sleep)
    ts = [threading.Thread(target=rl.acquire) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(waits) == [float(i) for i in range(1, 8)]  # every caller got a distinct slot


def test_rate_limiter_catches_up_after_idle():
    now = [0.0]
    rl = RateLimiter(60, clock=lambda: now[0], sleep=lambda s: None)
    rl.acquire()
    now[0] = 50.0  # long idle: no accumulated burst credit, next call is immediate
    assert rl.acquire() == 0.0
    assert rl.acquire() == 1.0


def test_throttled_provider_forwards_and_limits():
    seen = []
    rl = RateLimiter(60, clock=lambda: 0.0, sleep=seen.append)
    tp = ThrottledProvider(ScriptedProvider(["a", "b"], name="n", model="m"), rl)
    assert (tp.name, tp.model) == ("n", "m")
    tp.complete([])
    tp.complete([])
    assert seen == [1.0]


# --- parallel harness -------------------------------------------------------
class _Tracking:
    """Provider that records how many calls are in flight at once."""

    peak = 0
    live = 0
    lock = threading.Lock()

    def __init__(self):
        self.inner = MockAgentProvider()
        self.name, self.model = "tracked", "t"

    def complete(self, messages, temperature=0.0):
        cls = _Tracking
        with cls.lock:
            cls.live += 1
            cls.peak = max(cls.peak, cls.live)
        try:
            time.sleep(0.02)
            return self.inner.complete(messages, temperature)
        finally:
            with cls.lock:
                cls.live -= 1


def test_parallel_matches_sequential_and_overlaps():
    tasks = get_suite("basic")
    seq = EvalHarness(SqliteStore(":memory:")).run(tasks, [_cfg("m")], trials=2)
    _Tracking.peak = 0
    par = EvalHarness(SqliteStore(":memory:")).run(tasks, [EvalConfig("m", _Tracking)], trials=2, workers=4)
    key = lambda r: (r["config"], r["task_id"], r["trial"])  # noqa: E731
    assert [key(r) for r in par.rows] == [key(r) for r in seq.rows]  # deterministic order
    assert [r["answer"] for r in par.rows] == [r["answer"] for r in seq.rows]
    assert _Tracking.peak >= 2


def test_parallel_progress_and_all_persisted():
    store, msgs = SqliteStore(":memory:"), []
    rep = EvalHarness(store).run(get_suite("basic"), [_cfg("m")], workers=4, progress=msgs.append)
    assert len(msgs) == len(rep.rows) == len(store.get_eval_rows(rep.eval_id))
    assert all(m.startswith("[m]") for m in msgs)


def test_rpm_is_enforced_by_harness():
    t0 = time.perf_counter()
    rep = EvalHarness(SqliteStore(":memory:")).run(get_suite("basic")[:1], [_cfg("m")], rpm=600)  # 0.1s spacing
    llm_calls = rep.rows[0]["metrics"]["llm_calls"]
    assert time.perf_counter() - t0 >= (llm_calls - 1) * 0.1 * 0.9


def test_rpm_limit_is_per_config_and_shared_across_workers():
    t0 = time.perf_counter()
    h = EvalHarness(SqliteStore(":memory:"))
    rep = h.run(get_suite("basic")[:4], [_cfg("m")], workers=4, rpm=1200)  # 0.05s spacing, 4 parallel runs
    calls = sum(r["metrics"]["llm_calls"] for r in rep.rows)
    assert time.perf_counter() - t0 >= (calls - 1) * 0.05 * 0.9  # the limit holds across threads


def test_sleep_between_runs(monkeypatch):
    slept = []
    monkeypatch.setattr("agentlab.eval.harness.time.sleep", slept.append)
    EvalHarness(SqliteStore(":memory:")).run(get_suite("basic")[:3], [_cfg("m")], sleep_between=0.5)
    assert [s for s in slept if s == 0.5] == [0.5, 0.5, 0.5]  # time is global: ignore the mock's own 1ms sleeps
