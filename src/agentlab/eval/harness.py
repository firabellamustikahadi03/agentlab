"""Eval harness: tasks x configs x trials -> metrics, comparison report, regression gate."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean

from ..llm.base import Provider
from ..runtime import AgentRuntime, Budget
from ..tools import ToolRegistry, default_registry
from ..tracing import SqliteStore
from .scorers import score
from .suites import EvalTask


@dataclass
class EvalConfig:
    name: str
    make_provider: Callable[[], Provider]  # fresh provider per run keeps stateful mocks honest
    use_critic: bool = True
    budget: Budget = field(default_factory=Budget)


@dataclass
class EvalReport:
    eval_id: str
    rows: list[dict]

    def configs(self) -> list[str]:
        return list(dict.fromkeys(r["config"] for r in self.rows))

    def summary(self) -> dict[str, dict]:
        out = {}
        for c in self.configs():
            rs = [r for r in self.rows if r["config"] == c]
            n_tools = sum(r["metrics"]["tool_calls"] for r in rs)
            out[c] = {
                "runs": len(rs),
                "success_rate": mean(r["passed"] for r in rs),
                "avg_score": mean(r["score"] for r in rs),
                "avg_latency_s": mean(r["metrics"]["latency_s"] for r in rs),
                "avg_tokens": mean(r["metrics"]["tokens"] for r in rs),
                "total_cost_usd": sum(r["metrics"]["cost_usd"] for r in rs),
                "avg_llm_calls": mean(r["metrics"]["llm_calls"] for r in rs),
                "tool_error_rate": (sum(r["metrics"]["tool_errors"] for r in rs) / n_tools) if n_tools else 0.0,
                "replan_rate": mean(1.0 if r["metrics"]["replans"] else 0.0 for r in rs),
                "agent_failures": sum(1 for r in rs if r["status"] in ("failed", "budget_exceeded")),
            }
        return out

    def task_matrix(self) -> dict[str, dict[str, str]]:
        m: dict[str, dict[str, list[int]]] = {}
        for r in self.rows:
            m.setdefault(r["task_id"], {}).setdefault(r["config"], []).append(r["passed"])
        return {t: {c: f"{sum(v)}/{len(v)}" for c, v in cs.items()} for t, cs in m.items()}

    def to_markdown(self) -> str:
        s, cfgs = self.summary(), self.configs()
        lines = [f"# Eval `{self.eval_id}`", "", "## Summary", ""]
        lines.append(
            "| config | runs | success | avg tokens | avg latency (s) | cost ($) "
            "| llm calls | tool err | replan | agent fail |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in cfgs:
            x = s[c]
            lines.append(
                f"| {c} | {x['runs']} | {x['success_rate']:.0%} | {x['avg_tokens']:.0f} | {x['avg_latency_s']:.2f} "
                f"| {x['total_cost_usd']:.4f} | {x['avg_llm_calls']:.1f} | {x['tool_error_rate']:.0%} "
                f"| {x['replan_rate']:.0%} | {x['agent_failures']} |"
            )
        lines += [
            "", "## Per task (passed/trials)", "",
            "| task | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs),
        ]
        for t, row in self.task_matrix().items():
            lines.append(f"| {t} | " + " | ".join(row.get(c, "-") for c in cfgs) + " |")
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        payload = {"eval_id": self.eval_id, "summary": self.summary(), "rows": self.rows}
        return json.dumps(payload, indent=2, default=str)

    def regressions(self, baseline: dict, max_drop: float = 0.05) -> list[str]:
        """Compare against a saved report's summary; flag configs whose success rate dropped."""
        bad = []
        cur = self.summary()
        for c, b in baseline.get("summary", {}).items():
            if c in cur and cur[c]["success_rate"] < b["success_rate"] - max_drop:
                bad.append(f"{c}: success {b['success_rate']:.0%} -> {cur[c]['success_rate']:.0%}")
        return bad


class EvalHarness:
    def __init__(self, store: SqliteStore | None = None, tools: ToolRegistry | None = None):
        self.store = store or SqliteStore(":memory:")
        self.tools = tools or default_registry()

    def run(
        self,
        tasks: list[EvalTask],
        configs: list[EvalConfig],
        trials: int = 1,
        sleep_between: float = 0.0,  # throttle for free-tier rate limits
        progress: Callable[[str], None] | None = None,
    ) -> EvalReport:
        eval_id = "ev-" + uuid.uuid4().hex[:8]
        rows = []
        for cfg in configs:
            for task in tasks:
                for trial in range(trials):
                    rt = AgentRuntime(
                        cfg.make_provider(), self.tools, self.store, cfg.budget, use_critic=cfg.use_critic
                    )
                    res = rt.run(task.prompt)
                    spans = self.store.get_spans(res.run_id)
                    passed, sc, detail = score(task.checks, res.answer, spans)
                    if res.status in ("failed", "budget_exceeded"):
                        passed, sc = False, 0.0
                        detail = f"{res.status}: {res.error}; " + detail
                    self.store.save_eval_result(
                        eval_id, cfg.name, task.id, trial, res.run_id, passed, sc, res.metrics, detail
                    )
                    rows.append(
                        {
                            "config": cfg.name, "task_id": task.id, "trial": trial, "run_id": res.run_id,
                            "status": res.status, "passed": int(passed), "score": sc,
                            "answer": res.answer, "metrics": res.metrics, "detail": detail,
                        }
                    )
                    if progress:
                        progress(f"[{cfg.name}] {task.id}#{trial} {'PASS' if passed else 'FAIL'} ({res.status})")
                    if sleep_between:
                        time.sleep(sleep_between)
        return EvalReport(eval_id, rows)
