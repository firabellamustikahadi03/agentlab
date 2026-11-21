import json

import pytest

from agentlab import AgentRuntime, SqliteStore, default_registry
from agentlab.eval import EvalConfig, EvalHarness, check, get_suite
from agentlab.jsonutil import extract_json
from agentlab.llm import MockAgentProvider, ScriptedProvider
from agentlab.runtime import Budget
from agentlab.tools import run_python, safe_eval
from agentlab.tracing import ReplayProvider


# --- json -------------------------------------------------------------------
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('blah {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json("no json here") is None


# --- tools ------------------------------------------------------------------
def test_safe_eval_blocks_code():
    assert safe_eval("2^10 - 24") == 1000
    for bad in ["__import__('os').system('id')", "open('x')", "().__class__", "9**9**9"]:
        with pytest.raises(Exception):
            safe_eval(bad)


def test_python_sandbox():
    assert run_python("print(2+2)") == "4"
    with pytest.raises(RuntimeError):
        run_python("import os; os.environ['HOME']")  # env is empty
    with pytest.raises(TimeoutError):
        run_python("while True: pass", timeout_s=1)


def test_tool_errors_become_observations():
    reg = default_registry()
    assert reg.get("calculator").run({"expression": "1/0"}).ok is False
    assert "missing" in reg.get("calculator").run({}).error
    assert "unknown" in reg.get("calculator").run({"expression": "1", "x": 1}).error


# --- runtime ----------------------------------------------------------------
def test_basic_run_and_trace():
    store = SqliteStore(":memory:")
    res = AgentRuntime(MockAgentProvider(), store=store).run("What is 17*23+5?")
    assert res.ok and res.answer == "396"
    kinds = {s["kind"] for s in store.get_spans(res.run_id)}
    assert {"run", "plan", "step", "llm", "tool", "critic"} <= kinds
    assert res.metrics["tool_calls"] == 1


def test_critic_triggers_replan():
    res = AgentRuntime(MockAgentProvider(fail_first=True), store=SqliteStore(":memory:")).run("What is 7*8*9?")
    assert res.ok and res.answer == "504" and res.metrics["replans"] == 1


def test_unverified_when_replans_exhausted():
    p = ScriptedProvider(
        ['{"steps":["a"]}', '{"final":"x"}', '{"pass":false,"issues":"bad"}'] * 1
        + ['{"steps":["a"]}', '{"final":"y"}', '{"pass":false,"issues":"still bad"}']
    )
    res = AgentRuntime(p, store=SqliteStore(":memory:")).run("t")
    assert res.status == "unverified" and res.answer == "y"


def test_json_repair_and_fallback_plan():
    p = ScriptedProvider(["not json", "still not json", '{"final":"42"}', '{"pass":true,"issues":""}'])
    res = AgentRuntime(p, store=SqliteStore(":memory:")).run("t")
    assert res.ok and res.answer == "42" and res.plan == ["Solve the task: t"]


def test_unknown_tool_is_recoverable():
    p = ScriptedProvider([
        '{"steps":["s"]}', '{"action":{"tool":"nope","args":{}}}', '{"final":"done"}', '{"pass":true,"issues":""}',
    ])
    res = AgentRuntime(p, store=SqliteStore(":memory:")).run("t")
    assert res.ok and res.metrics["tool_errors"] == 1


def test_budget_guard():
    p = MockAgentProvider()
    res = AgentRuntime(p, store=SqliteStore(":memory:"), budget=Budget(max_llm_calls=1)).run("What is 1+1?")
    assert res.status == "budget_exceeded"


class CrashAfter:
    """Provider that raises on the n-th call, then delegates: simulates a process crash."""

    name, model = "crash", "crash-1"

    def __init__(self, inner, n):
        self.inner, self.n, self.i = inner, n, 0

    def complete(self, messages, temperature=0.0):
        self.i += 1
        if self.i == self.n:
            raise ConnectionError("boom")
        return self.inner.complete(messages, temperature)


def test_resume_after_crash():
    store = SqliteStore(":memory:")
    task = "What is 12*12?"
    res = AgentRuntime(CrashAfter(MockAgentProvider(), 4), store=store).run(task)  # crash in critic
    assert res.status == "failed"
    assert store.get_run(res.run_id)["status"] == "failed"
    res2 = AgentRuntime(MockAgentProvider(), store=store).resume(res.run_id)
    assert res2.ok and res2.answer == "144"
    # the plan and step result came from the checkpoint, not recomputed
    assert res2.metrics["llm_calls"] == 4  # 3 before the crash + only the critic call after resume
    spans = store.get_spans(res.run_id)
    assert sum(s["kind"] == "plan" for s in spans) == 1 and sum(s["kind"] == "step" for s in spans) == 1


def test_replay_is_deterministic():
    store = SqliteStore(":memory:")
    first = AgentRuntime(MockAgentProvider(fail_first=True), store=store).run("What is 3*3?")
    rp = ReplayProvider(store, first.run_id)
    again = AgentRuntime(rp, store=store).run("What is 3*3?")
    assert again.answer == first.answer and again.metrics["replans"] == first.metrics["replans"]
    assert rp.divergences == []


# --- eval -------------------------------------------------------------------
def test_scorers():
    assert check({"type": "numeric", "value": 1000}, "The answer is 1,000.", [])[0]
    assert not check({"type": "numeric", "value": 5}, "no numbers", [])[0]
    assert check({"type": "tool_used", "tool": "python"}, "", [{"kind": "tool", "name": "python"}])[0]
    assert check({"type": "python_assert", "code": "assert answer == '4'"}, "4", [])[0]


def test_harness_and_regression_gate():
    h = EvalHarness(SqliteStore(":memory:"))
    cfgs = [
        EvalConfig("mock", lambda: MockAgentProvider()),
        EvalConfig("mock-flaky", lambda: MockAgentProvider(fail_first=True)),
    ]
    rep = h.run(get_suite("basic"), cfgs)
    s = rep.summary()
    assert s["mock"]["success_rate"] == 1.0
    assert s["mock-flaky"]["replan_rate"] == 1.0
    assert "| mock |" in rep.to_markdown()
    base = json.loads(rep.to_json())
    assert rep.regressions(base) == []
    worse = h.run(get_suite("reasoning"), [EvalConfig("mock", lambda: MockAgentProvider())])
    assert worse.regressions(base)  # reasoning suite drops the mock's success rate
