"""Eval harness: tasks x configs x trials -> metrics, comparison report, regression gate.

Runs can execute in parallel (thread pool) behind a per-config rate limiter, which is what keeps a
multi-trial eval inside a free tier's requests-per-minute quota.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import mean

from ..llm.base import Provider
from ..llm.throttle import RateLimiter, ThrottledProvider
from ..prompts import PROMPT_SETS, PromptSet
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
    prompts: PromptSet | None = None  # None = the default prompts


def variant_configs(base: EvalConfig, variants: list[str]) -> list[EvalConfig]:
    """Expand one config into prompt/critic variants so they can be compared side by side.

    A variant is `default`, `no-critic`, or a PromptSet name (`concise`, `careful`); join several with
    `+` (e.g. `careful+no-critic`). The result is named `<base>` for `default`, else `<base>+<variant>`.
    """
    out = []
    for v in variants:
        parts = [p for p in v.split("+") if p]
        cfg = EvalConfig(
            name=base.name if parts == ["default"] else f"{base.name}+{v}",
            make_provider=base.make_provider, use_critic=base.use_critic, budget=base.budget, prompts=base.prompts,
        )
        for part in parts:
            if part == "default":
                continue
            if part == "no-critic":
                cfg.use_critic = False
            elif part in PROMPT_SETS:
                cfg.prompts = PROMPT_SETS[part]
            else:
                raise ValueError(f"unknown variant {part!r}; use default, no-critic or {sorted(PROMPT_SETS)}")
        out.append(cfg)
    return out


@dataclass
class EvalReport:
    eval_id: str
    rows: list[dict]

    @classmethod
    def from_store(cls, store: SqliteStore, eval_id: str) -> EvalReport:
        if store.get_eval(eval_id) is None:
            raise KeyError(f"eval {eval_id} not found")
        return cls(eval_id, store.get_eval_rows(eval_id))

    def configs(self) -> list[str]:
        return list(dict.fromkeys(r["config"] for r in self.rows))

    def summary(self) -> dict[str, dict]:
        out = {}
        for c in self.configs():
            rs = [r for r in self.rows if r["config"] == c]
            n_tools = sum(r["metrics"]["tool_calls"] for r in rs)
            total_cost = sum(r["metrics"]["cost_usd"] for r in rs)
            out[c] = {
                "runs": len(rs),
                "success_rate": mean(r["passed"] for r in rs),
                "avg_score": mean(r["score"] for r in rs),
                "avg_latency_s": mean(r["metrics"]["latency_s"] for r in rs),
                "avg_tokens": mean(r["metrics"]["tokens"] for r in rs),
                "total_cost_usd": total_cost,
                "avg_cost_usd": total_cost / len(rs),
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

    def task_metrics(self) -> dict[str, dict[str, dict]]:
        """task -> config -> averages over trials: pass rate, tokens, latency, cost."""
        grouped: dict[str, dict[str, list[dict]]] = {}
        for r in self.rows:
            grouped.setdefault(r["task_id"], {}).setdefault(r["config"], []).append(r)
        return {
            t: {
                c: {
                    "pass_rate": mean(r["passed"] for r in rs),
                    "tokens": mean(r["metrics"]["tokens"] for r in rs),
                    "latency_s": mean(r["metrics"]["latency_s"] for r in rs),
                    "cost_usd": mean(r["metrics"]["cost_usd"] for r in rs),
                    "llm_calls": mean(r["metrics"]["llm_calls"] for r in rs),
                }
                for c, rs in cs.items()
            }
            for t, cs in grouped.items()
        }

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
        lines += [
            "", "## Per task cost, latency and tokens (mean over trials)", "",
            "| task | config | pass | tokens | latency (s) | cost ($) | llm calls |", "|---|---|---|---|---|---|---|",
        ]
        for t, by_cfg in self.task_metrics().items():
            for c in cfgs:
                if c in by_cfg:
                    m = by_cfg[c]
                    lines.append(
                        f"| {t} | {c} | {m['pass_rate']:.0%} | {m['tokens']:.0f} | {m['latency_s']:.2f} "
                        f"| {m['cost_usd']:.5f} | {m['llm_calls']:.1f} |"
                    )
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
    def __init__(
        self, store: SqliteStore | None = None, tools: ToolRegistry | None = None, judge: Provider | None = None
    ):
        self.store = store or SqliteStore(":memory:")
        self.tools = tools or default_registry()
        self.judge = judge  # provider used by {"type": "judge"} checks

    def run(
        self,
        tasks: list[EvalTask],
        configs: list[EvalConfig],
        trials: int = 1,
        sleep_between: float = 0.0,  # pause after each run (simple throttle, per worker)
        progress: Callable[[str], None] | None = None,
        workers: int = 1,  # >1 runs eval jobs on a thread pool
        rpm: float | None = None,  # max LLM requests per minute, enforced per config (free-tier quotas)
        suite: str = "",  # recorded with the eval so `eval-list` can show it
    ) -> EvalReport:
        eval_id = "ev-" + uuid.uuid4().hex[:8]
        meta = {
            "configs": [c.name for c in configs], "tasks": len(tasks), "trials": trials,
            "workers": workers, "rpm": rpm,
        }
        self.store.save_eval_run(eval_id, suite, meta)
        limiters = {c.name: RateLimiter(rpm) for c in configs} if rpm else {}
        judge = ThrottledProvider(self.judge, RateLimiter(rpm)) if (rpm and self.judge) else self.judge
        say_lock = threading.Lock()

        def say(msg: str) -> None:
            if progress:
                with say_lock:
                    progress(msg)

        def job(cfg: EvalConfig, task: EvalTask, trial: int) -> dict:
            provider = cfg.make_provider()
            if cfg.name in limiters:
                provider = ThrottledProvider(provider, limiters[cfg.name])
            tools = task.tools() if task.tools else self.tools  # fresh per run: tools may be stateful
            rt = AgentRuntime(
                provider, tools, self.store, cfg.budget, use_critic=cfg.use_critic, prompt_set=cfg.prompts
            )
            res = rt.run(task.prompt)
            spans = self.store.get_spans(res.run_id)
            passed, sc, detail = score(task.checks, res.answer, spans, judge, task.prompt)
            if res.status in ("failed", "budget_exceeded"):
                passed, sc = False, 0.0
                detail = f"{res.status}: {res.error}; " + detail
            self.store.save_eval_result(
                eval_id, cfg.name, task.id, trial, res.run_id, passed, sc, res.metrics, detail,
                status=res.status, answer=res.answer,
            )
            say(f"[{cfg.name}] {task.id}#{trial} {'PASS' if passed else 'FAIL'} ({res.status})")
            if sleep_between:
                time.sleep(sleep_between)
            return {
                "config": cfg.name, "task_id": task.id, "trial": trial, "run_id": res.run_id,
                "status": res.status, "passed": int(passed), "score": sc,
                "answer": res.answer, "metrics": res.metrics, "detail": detail,
            }

        jobs = [(cfg, task, trial) for cfg in configs for task in tasks for trial in range(trials)]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(job, *j) for j in jobs]
                rows = [f.result() for f in futures]  # submission order => deterministic report
        else:
            rows = [job(*j) for j in jobs]
        return EvalReport(eval_id, rows)
