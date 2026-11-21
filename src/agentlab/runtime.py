"""AgentRuntime: planner -> executor (tool loop) -> critic, with replanning,
budget guards, checkpoints after every step, and full span tracing."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .jsonutil import extract_json
from .llm.base import Message, Provider
from .tools import ToolRegistry, default_registry
from .tracing import SqliteStore, Tracer


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_llm_calls: int = 40
    max_tokens: int = 60_000
    max_seconds: float = 180.0
    max_tool_iters: int = 6  # per step
    max_replans: int = 1


@dataclass
class RunResult:
    run_id: str
    status: str  # success | unverified | failed | budget_exceeded
    answer: str | None
    plan: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success"


def _new_state(task: str) -> dict:
    return {
        "task": task, "plan": [], "step_idx": 0, "step_results": [], "replans": 0,
        "feedback": "", "answer": None, "status": "running", "error": "", "critic_pass": None,
        "elapsed": 0.0,
        "usage": {"llm_calls": 0, "tokens": 0, "cost_usd": 0.0, "tool_calls": 0, "tool_errors": 0},
    }


class AgentRuntime:
    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        store: SqliteStore | None = None,
        budget: Budget | None = None,
        use_critic: bool = True,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.tools = tools or default_registry()
        self.store = store or SqliteStore(":memory:")
        self.budget = budget or Budget()
        self.use_critic = use_critic
        self.temperature = temperature

    # public API ---------------------------------------------------------
    def run(self, task: str, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:10]
        self.store.upsert_run(
            run_id, task=task, provider=self.provider.name, model=self.provider.model,
            status="running", started=time.time(),
        )
        return self._drive(run_id, _new_state(task), root="run")

    def resume(self, run_id: str) -> RunResult:
        state = self.store.latest_checkpoint(run_id)
        if state is None:
            raise KeyError(f"no checkpoint for run {run_id}")
        if state["status"] != "running":
            return self._result(run_id, state)
        state["error"] = ""
        self.store.upsert_run(run_id, status="running")
        return self._drive(run_id, state, root="resume")

    # driver -------------------------------------------------------------
    def _drive(self, run_id: str, state: dict, root: str) -> RunResult:
        existing = self.store.get_spans(run_id)
        tr = Tracer(self.store, run_id, start_seq=(max((s["seq"] for s in existing), default=-1) + 1))
        seg_start = time.time()
        self._ckpt_seq = len(existing)
        try:
            with tr.span(root, state["task"], {"task": state["task"]}) as rs:
                try:
                    self._loop(tr, run_id, state, seg_start)
                except BudgetExceeded as e:
                    state["status"], state["error"] = "budget_exceeded", str(e)
                    state["answer"] = state["answer"] or (state["step_results"][-1] if state["step_results"] else None)
                except Exception as e:
                    # keep state "running" so resume() can pick the run up after a crash
                    state["error"] = f"{type(e).__name__}: {e}"
                    state["crashed"] = True
                    rs.set(crashed=True)
                state["elapsed"] += time.time() - seg_start
                rs.set(status=state["status"], **self._metrics(state))
                rs.d["output"] = {"answer": state["answer"], "error": state["error"]}
        finally:
            self._checkpoint(run_id, state)
            crashed = state.pop("crashed", False)
            self.store.upsert_run(
                run_id, status="failed" if crashed else state["status"], ended=time.time(),
                meta=self._metrics(state),
            )
        res = self._result(run_id, state)
        if state.get("error") and state["status"] == "running":
            res.status = "failed"
        return res

    def _loop(self, tr: Tracer, run_id: str, state: dict, seg_start: float) -> None:
        self._seg_start = seg_start
        self._state = state
        task = state["task"]
        while True:
            if not state["plan"]:
                state["plan"] = self._plan(tr, task, state["feedback"])
                self._checkpoint(run_id, state)
            while state["step_idx"] < len(state["plan"]):
                i = state["step_idx"]
                with tr.span("step", f"step {i + 1}", {"step": state["plan"][i]}) as sp:
                    res = self._execute_step(tr, task, state, i)
                    sp.d["output"] = {"result": res}
                state["step_results"].append(res)
                state["step_idx"] += 1
                self._checkpoint(run_id, state)
            answer = state["step_results"][-1] if state["step_results"] else ""
            state["answer"] = answer
            if not self.use_critic:
                state["status"] = "success"
                return
            verdict = self._critic(tr, task, answer)
            state["critic_pass"] = verdict["pass"]
            if verdict["pass"]:
                state["status"] = "success"
                return
            if state["replans"] >= self.budget.max_replans:
                state["status"] = "unverified"
                return
            state.update(replans=state["replans"] + 1, feedback=verdict["issues"], plan=[], step_idx=0, step_results=[])
            self._checkpoint(run_id, state)

    # LLM plumbing -------------------------------------------------------
    def _guard(self) -> None:
        u, b, s = self._state["usage"], self.budget, self._state
        if u["llm_calls"] >= b.max_llm_calls:
            raise BudgetExceeded(f"max_llm_calls={b.max_llm_calls}")
        if u["tokens"] >= b.max_tokens:
            raise BudgetExceeded(f"max_tokens={b.max_tokens}")
        if s["elapsed"] + (time.time() - self._seg_start) > b.max_seconds:
            raise BudgetExceeded(f"max_seconds={b.max_seconds}")

    def _llm(self, tr: Tracer, role: str, messages: list[Message]) -> str:
        self._guard()
        with tr.span("llm", role, {"n_messages": len(messages), "last": messages[-1].content[:500]}) as sp:
            r = self.provider.complete(messages, temperature=self.temperature)
            sp.set(
                prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                latency_ms=round(r.latency_ms, 2), cost_usd=r.cost_usd, model=r.model,
            )
            sp.d["output"] = {"content": r.content}
        u = self._state["usage"]
        u["llm_calls"] += 1
        u["tokens"] += r.usage.total
        u["cost_usd"] += r.cost_usd
        return r.content

    def _llm_json(self, tr: Tracer, role: str, messages: list[Message]) -> dict | None:
        """Call the model; if the reply is not JSON, ask once to repair it."""
        text = self._llm(tr, role, messages)
        obj = extract_json(text)
        if obj is not None:
            return obj
        repair = messages + [
            Message("assistant", text),
            Message("user", "Your last reply was not valid JSON. Reply again with ONLY the JSON object."),
        ]
        return extract_json(self._llm(tr, f"{role}:repair", repair))

    # roles --------------------------------------------------------------
    def _plan(self, tr: Tracer, task: str, feedback: str) -> list[str]:
        with tr.span("plan", "plan", {"task": task, "feedback": feedback}) as sp:
            user = f"TASK: {task}"
            if feedback:
                user += f"\nPREVIOUS ATTEMPT FAILED, CRITIC SAID: {feedback}\nMake a better plan."
            obj = self._llm_json(
                tr, "planner",
                [Message("system", prompts.PLANNER.format(tools=self.tools.describe())), Message("user", user)],
            )
            steps = [str(s) for s in (obj or {}).get("steps", []) if str(s).strip()][:6]
            if not steps:  # planner failed: degrade gracefully to a single-step plan
                steps = [f"Solve the task: {task}"]
                sp.set(fallback_plan=True)
            sp.d["output"] = {"steps": steps}
            return steps

    def _execute_step(self, tr: Tracer, task: str, state: dict, i: int) -> str:
        done = "\n".join(f"- step {j + 1} result: {r}" for j, r in enumerate(state["step_results"]))
        user = (
            f"TASK: {task}\nPLAN:\n" + "\n".join(f"{j + 1}. {s}" for j, s in enumerate(state["plan"]))
            + f"\nCURRENT STEP: {i + 1}. {state['plan'][i]}"
            + (f"\nPREVIOUS RESULTS:\n{done}" if done else "")
        )
        msgs = [Message("system", prompts.EXECUTOR.format(tools=self.tools.describe())), Message("user", user)]
        for _ in range(self.budget.max_tool_iters):
            obj = self._llm_json(tr, "executor", msgs)
            if obj is None:
                msgs += [
                    Message("assistant", "(invalid)"),
                    Message("user", "OBSERVATION: invalid reply, send valid JSON."),
                ]
                continue
            if "final" in obj:
                return str(obj["final"])
            act = obj.get("action")
            if isinstance(act, dict) and act.get("tool"):
                obs = self._call_tool(tr, str(act["tool"]), act.get("args") or {})
                msgs += [Message("assistant", _dump(obj)), Message("user", f"OBSERVATION: {obs}")]
            else:
                msgs += [
                    Message("assistant", _dump(obj)),
                    Message("user", "OBSERVATION: need an action or a final key."),
                ]
        return "(step incomplete: tool iteration limit reached)"

    def _call_tool(self, tr: Tracer, name: str, args: dict) -> str:
        u = self._state["usage"]
        u["tool_calls"] += 1
        with tr.span("tool", name, args) as sp:
            tool = self.tools.get(name)
            if tool is None:
                res_ok, text = False, f"unknown tool '{name}'. Available: {self.tools.names()}"
            else:
                t0 = time.perf_counter()
                r = tool.run(args if isinstance(args, dict) else {})
                sp.set(latency_ms=round((time.perf_counter() - t0) * 1000, 2))
                res_ok, text = r.ok, r.render()
            if not res_ok:
                u["tool_errors"] += 1
                sp.d["status"] = "error"
            sp.d["output"] = {"ok": res_ok, "text": text}
            return text if res_ok else (text if text.startswith("ERROR") else f"ERROR: {text}")

    def _critic(self, tr: Tracer, task: str, answer: str) -> dict:
        with tr.span("critic", "critic", {"answer": answer}) as sp:
            obj = self._llm_json(
                tr, "critic",
                [Message("system", prompts.CRITIC), Message("user", f"TASK: {task}\nANSWER: {answer}")],
            )
            if obj is None or "pass" not in obj:  # unparseable critic must not block a result
                v = {"pass": True, "issues": "critic unavailable"}
                sp.set(critic_unavailable=True)
            else:
                v = {"pass": bool(obj["pass"]), "issues": str(obj.get("issues", ""))}
            sp.d["output"] = v
            return v

    # bookkeeping --------------------------------------------------------
    def _checkpoint(self, run_id: str, state: dict) -> None:
        self._ckpt_seq += 1
        self.store.save_checkpoint(run_id, self._ckpt_seq, state)

    def _metrics(self, state: dict) -> dict:
        u = state["usage"]
        return {
            "llm_calls": u["llm_calls"], "tokens": u["tokens"], "cost_usd": round(u["cost_usd"], 6),
            "tool_calls": u["tool_calls"], "tool_errors": u["tool_errors"],
            "steps": len(state["plan"]), "replans": state["replans"],
            "latency_s": round(state["elapsed"], 4), "critic_pass": state["critic_pass"],
        }

    def _result(self, run_id: str, state: dict) -> RunResult:
        return RunResult(run_id, state["status"], state["answer"], state["plan"], self._metrics(state), state["error"])


def _dump(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
