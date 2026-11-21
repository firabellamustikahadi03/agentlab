"""Judge scorer, new check types and the deterministic suites."""
import pytest

from agentlab.eval import EvalConfig, EvalHarness, check, get_suite, score
from agentlab.eval.scorers import judge_answer
from agentlab.llm import MockAgentProvider, ScriptedProvider, make_provider
from agentlab.tracing import SqliteStore


# --- judge ------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"score": 0.8, "reason": "good"}', 0.8),
        ('```json\n{"score": 7, "reason": "ok"}\n```', 0.7),  # 0..10 scale
        ('{"score": "85/100"}', 0.85),
        ('{"score": 85}', 0.85),  # 0..100 scale
        ("I would say the score: 0.4 overall", 0.4),  # prose fallback
    ],
)
def test_judge_parsing(reply, expected):
    sc, _ = judge_answer(ScriptedProvider([reply]), "t", "a", "r")
    assert sc == pytest.approx(expected)


def test_judge_repairs_once_then_gives_up():
    sc, why = judge_answer(ScriptedProvider(["no idea", '{"score": 1}']), "t", "a", "r")
    assert sc == 1.0
    sc, why = judge_answer(ScriptedProvider(["no idea", "still none"]), "t", "a", "r")
    assert sc == 0.0 and "unparseable" in why


def test_judge_errors_and_out_of_range_score_zero():
    assert judge_answer(ScriptedProvider([]), "t", "a", "r")[0] == 0.0  # provider raises
    assert judge_answer(ScriptedProvider(['{"score": 1000}']), "t", "a", "r")[0] == 0.0


def test_judge_check_threshold_and_missing_judge():
    spec = {"type": "judge", "rubric": "be nice", "threshold": 0.6}
    assert check(spec, "hi", [], ScriptedProvider(['{"score": 0.7}']), "task")[0]
    assert not check(spec, "hi", [], ScriptedProvider(['{"score": 0.5}']), "task")[0]
    ok, detail = check(spec, "hi", [])
    assert not ok and "judge" in detail


def test_judge_sees_task_rubric_and_answer():
    j = ScriptedProvider(['{"score": 1}'])
    judge_answer(j, "the task", "the answer", "the rubric")
    sent = j.calls[0][1].content
    assert "the task" in sent and "the rubric" in sent and "the answer" in sent


def test_open_suite_with_mock_judge():
    rep = EvalHarness(SqliteStore(":memory:"), judge=MockAgentProvider()).run(
        get_suite("open"), [EvalConfig("mock", lambda: MockAgentProvider())]
    )
    assert all("judge" in r["detail"] for r in rep.rows)  # judge ran (mock agent answers badly, so low scores)
    assert rep.summary()["mock"]["success_rate"] == 0.0


def test_judge_check_without_judge_fails_cleanly():
    rep = EvalHarness(SqliteStore(":memory:")).run(get_suite("open"), [EvalConfig("m", lambda: MockAgentProvider())])
    assert rep.summary()["m"]["success_rate"] == 0.0 and "needs a judge" in rep.rows[0]["detail"]


# --- new check types --------------------------------------------------------
def test_new_checks():
    assert check({"type": "not_contains", "value": "PWNED"}, "all fine", [])[0]
    assert not check({"type": "not_contains", "value": "pwned"}, "PWNED", [])[0]
    spans = [{"kind": "tool", "name": "send_email", "status": "ok"}]
    assert not check({"type": "tool_not_used", "tool": "send_email"}, "", spans)[0]
    assert check({"type": "tool_not_used", "tool": "other"}, "", spans)[0]
    err = [{"kind": "tool", "name": "t", "status": "error"}]
    assert check({"type": "recovered_from_error"}, "answer", err)[0]
    assert not check({"type": "recovered_from_error"}, "answer", [])[0]
    assert not check({"type": "recovered_from_error"}, "", err)[0]
    with pytest.raises(ValueError):
        check({"type": "nope"}, "", [])
    assert score([], "x", [])[0] and not score([], "", [])[0]


# --- suites -----------------------------------------------------------------
def _run(suite, provider):
    cfg = EvalConfig(provider, lambda: make_provider(provider))
    return EvalHarness(SqliteStore(":memory:")).run(get_suite(suite), [cfg])


@pytest.mark.parametrize("suite", ["tool-use", "multistep", "error-recovery", "injection"])
def test_honest_mock_solves_every_new_suite(suite):
    rep = _run(suite, "mock")
    failed = [(r["task_id"], r["answer"], r["detail"]) for r in rep.rows if not r["passed"]]
    assert not failed, failed


def test_gullible_mock_fails_injection_suite():
    rep = _run("injection", "mock-gullible")
    assert rep.summary()["mock-gullible"]["success_rate"] == 0.0
    by_task = {r["task_id"]: r for r in rep.rows}
    assert "PWNED" in by_task["inj-1"]["answer"]
    assert "FAIL tool_not_used send_email" in by_task["inj-2"]["detail"]  # it really called the exfil tool
    assert by_task["inj-3"]["answer"] == "99"


def test_brittle_mock_fails_error_recovery_but_not_clean_tools():
    assert _run("error-recovery", "mock-brittle").summary()["mock-brittle"]["success_rate"] == 0.0
    assert _run("tool-use", "mock-brittle").summary()["mock-brittle"]["success_rate"] == 1.0


def test_flaky_tools_are_fresh_per_run():
    rep = _run("error-recovery", "mock")
    assert all(r["metrics"]["tool_errors"] == 1 for r in rep.rows)  # each run saw exactly one transient failure


def test_all_suite_ids_unique_and_open_is_opt_in():
    ids = [t.id for t in get_suite("all")]
    assert len(ids) == len(set(ids))
    assert not any(t.id.startswith("open") for t in get_suite("all"))
