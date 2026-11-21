"""Seed a store with realistic offline data so the dashboard has something to show (no API keys needed).

What it creates: a handful of varied single runs (success, replan, tool use, a crash, a budget overrun) and two
evals of the same agent configs, where the second one "ships a bad prompt change" so `Compare` has a
regression to find.
"""
from __future__ import annotations

from collections.abc import Callable

from .eval import EvalConfig, EvalHarness, get_suite
from .eval.fixtures import shop_tools
from .llm import MockAgentProvider
from .runtime import AgentRuntime, Budget
from .tracing.base import Store

# (input, output) USD per 1M tokens: roughly a small hosted model, so costs are visible but tiny
_PRICE = (0.15, 0.60)


def _mock(**kw) -> MockAgentProvider:
    kw.setdefault("delay_s", 0.012)
    kw.setdefault("price_per_mtok", _PRICE)
    return MockAgentProvider(model="mock-small", **kw)


class _CrashesOnCall:
    """Provider that raises on its n-th call: simulates a network failure mid-run."""

    name, model = "mock", "mock-small"

    def __init__(self, n: int):
        self.inner, self.n, self.i = _mock(), n, 0

    def complete(self, messages, temperature=0.0):
        self.i += 1
        if self.i == self.n:
            raise ConnectionError("simulated network failure")
        return self.inner.complete(messages, temperature)


def seed_demo(store: Store, progress: Callable[[str], None] = lambda m: None, workers: int = 4) -> dict:
    """Populate `store`; returns ids of the two evals and the single runs."""
    runs: list[str] = []

    def run(task: str, provider, **kw) -> None:
        res = AgentRuntime(provider, kw.pop("tools", None), store, **kw).run(task)
        runs.append(res.run_id)
        progress(f"run {res.run_id}: {res.status}  {task[:50]}")

    h = EvalHarness(store)
    # Skip tasks that spawn the Python sandbox: process start-up can take seconds on locked-down machines, and
    # timeouts under parallel load would show up in the demo as regressions nobody planted.
    tasks = [t for t in get_suite("all") if "code" not in t.tags]

    def cfg(name: str, critic: bool = True, **mock_kw) -> EvalConfig:
        return EvalConfig(name, lambda: _mock(name=name, **mock_kw), use_critic=critic)

    progress("eval 1/2: baseline")
    baseline = h.run(
        tasks, [cfg("agent"), cfg("agent-flaky", fail_first=True), cfg("agent-no-critic", critic=False)],
        workers=workers, suite="all", progress=progress,
    )
    progress("eval 2/2: candidate (a prompt change that makes `agent` follow injected instructions)")
    candidate = h.run(
        tasks,
        [cfg("agent", gullible=True), cfg("agent-flaky", fail_first=True), cfg("agent-no-critic", critic=False),
         cfg("agent-brittle", retries=False)],
        workers=workers, suite="all", progress=progress,
    )

    # Single runs go last so they sit at the top of the (newest-first) run list.
    run("What is 17*23+5?", _mock())
    run("Use the unit_price tool for a pen and for a backpack, then use the calculator to get the total cost "
        "of 10 pens and 2 backpacks.", _mock(), tools=shop_tools())
    run("Start with 100. Divide by 4. Subtract 5. Multiply by 3. What is the result?", _mock())
    run("What is 1+1?", _mock(), budget=Budget(max_llm_calls=2))  # budget exceeded
    run("What is 12*12?", _CrashesOnCall(4))  # crashes in the critic: status failed, resumable
    run("What is 7*8*9?", _mock(fail_first=True))  # critic rejects the first answer -> replan
    return {"baseline": baseline.eval_id, "candidate": candidate.eval_id, "runs": runs}


def is_empty(store: Store) -> bool:
    return not store.list_runs(1) and not store.list_evals(1)
