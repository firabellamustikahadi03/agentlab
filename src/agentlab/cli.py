"""agentlab CLI: run | resume | eval | eval-list | eval-compare | traces list|show|replay"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .eval import EvalConfig, EvalHarness, EvalReport, compare_reports, get_suite, variant_configs
from .llm import make_provider
from .runtime import AgentRuntime
from .tracing import ReplayProvider, SqliteStore


def _fmt_span(s: dict, depth: int) -> str:
    dur = ((s["ended"] or s["started"]) - s["started"]) * 1000
    a = s["attrs"] or {}
    extra = ""
    if s["kind"] == "llm":
        extra = f" tok={a.get('prompt_tokens', 0)}+{a.get('completion_tokens', 0)}"
    mark = "x" if s["status"] == "error" else "-"
    body = json.dumps(s["output"], ensure_ascii=False, default=str)[:110] if s["output"] else ""
    return f"{'  ' * depth}{mark} {s['kind']}:{s['name'][:40]} {dur:.0f}ms{extra}  {body}"


def print_trace(store: SqliteStore, run_id: str) -> None:
    spans = store.get_spans(run_id)
    kids: dict = {}
    for s in spans:
        kids.setdefault(s["parent_id"], []).append(s)

    def walk(pid, depth):
        for s in kids.get(pid, []):
            print(_fmt_span(s, depth))
            walk(s["span_id"], depth + 1)

    walk(None, 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentlab")
    p.add_argument("--db", default="agentlab.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one task")
    r.add_argument("task")
    r.add_argument("--provider", default="mock")
    r.add_argument("--no-critic", action="store_true")

    rs = sub.add_parser("resume", help="resume a crashed run")
    rs.add_argument("run_id")
    rs.add_argument("--provider", default="mock")

    e = sub.add_parser("eval", help="evaluate providers on a suite")
    e.add_argument("--suite", default="basic")
    e.add_argument("--providers", nargs="+", default=["mock"])
    e.add_argument("--trials", type=int, default=1)
    e.add_argument("--sleep", type=float, default=0.0, help="seconds between runs (free-tier rate limits)")
    e.add_argument("--workers", type=int, default=1, help="run eval jobs in parallel on N threads")
    e.add_argument("--rpm", type=float, help="max LLM requests per minute, enforced per provider config")
    e.add_argument("--variants", nargs="+", default=["default"],
                   help="prompt/critic variants: default no-critic concise careful (join with +)")
    e.add_argument("--judge", help="provider spec used by judge-scored checks, e.g. groq or mock")
    e.add_argument("--out", help="write markdown report here")
    e.add_argument("--json", help="write JSON report here")
    e.add_argument("--baseline", help="JSON report to compare against (regression gate)")
    e.add_argument("--max-drop", type=float, default=0.05)

    sub.add_parser("eval-list", help="list saved eval runs")
    ec = sub.add_parser("eval-compare", help="diff two saved eval runs")
    ec.add_argument("eval_a")
    ec.add_argument("eval_b")
    ec.add_argument("--max-drop", type=float, default=0.05)
    ec.add_argument("--fail-on-regression", action="store_true", help="exit 2 if B is worse than A")

    t = sub.add_parser("traces", help="inspect traces")
    ts = t.add_subparsers(dest="tcmd", required=True)
    ts.add_parser("list")
    ts.add_parser("show").add_argument("run_id")
    ts.add_parser("replay").add_argument("run_id")

    a = p.parse_args(argv)
    store = SqliteStore(a.db)

    if a.cmd == "run":
        res = AgentRuntime(make_provider(a.provider), store=store, use_critic=not a.no_critic).run(a.task)
        print(f"run_id: {res.run_id}\nstatus: {res.status}\nplan:   {res.plan}\nanswer: {res.answer}")
        print("metrics:", json.dumps(res.metrics))
        return 0 if res.ok else 1

    if a.cmd == "resume":
        res = AgentRuntime(make_provider(a.provider), store=store).resume(a.run_id)
        print(f"status: {res.status}\nanswer: {res.answer}")
        return 0 if res.ok else 1

    if a.cmd == "eval":
        cfgs = []
        for spec in a.providers:
            cfgs += variant_configs(EvalConfig(spec, (lambda s=spec: make_provider(s))), a.variants)
        judge = make_provider(a.judge) if a.judge else None
        rep = EvalHarness(store, judge=judge).run(
            get_suite(a.suite), cfgs, a.trials, a.sleep, progress=lambda m: print(m, file=sys.stderr),
            workers=a.workers, rpm=a.rpm, suite=a.suite,
        )
        print(rep.to_markdown())
        if a.out:
            open(a.out, "w").write(rep.to_markdown())
        if a.json:
            open(a.json, "w").write(rep.to_json())
        if a.baseline:
            bad = rep.regressions(json.load(open(a.baseline)), a.max_drop)
            if bad:
                print("REGRESSIONS:\n  " + "\n  ".join(bad), file=sys.stderr)
                return 2
        return 0

    if a.cmd == "eval-list":
        for ev in store.list_evals():
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ev["created"]))
            ok = f"{ev['success']:.0%}" if ev["success"] is not None else "-"
            print(f"{ev['eval_id']}  {when}  {ev['suite'] or '-':<14} runs={ev['n']:<4} success={ok:<5}"
                  f" {','.join(ev['meta'].get('configs', []))}")
        return 0

    if a.cmd == "eval-compare":
        try:
            cmp = compare_reports(EvalReport.from_store(store, a.eval_a), EvalReport.from_store(store, a.eval_b))
        except KeyError as e:
            print(f"error: {e.args[0]}", file=sys.stderr)
            return 1
        print(cmp.to_markdown())
        bad = cmp.regressions(a.max_drop)
        if bad:
            print("REGRESSIONS:", file=sys.stderr)
            for line in bad:
                print("  " + line, file=sys.stderr)
            if a.fail_on_regression:
                return 2
        return 0

    if a.cmd == "traces":
        if a.tcmd == "list":
            for run in store.list_runs():
                print(f"{run['run_id']}  {run['status']:<16} {run['provider']:<10} {(run['task'] or '')[:60]}")
        elif a.tcmd == "show":
            print_trace(store, a.run_id)
        elif a.tcmd == "replay":
            orig = store.get_run(a.run_id)
            rp = ReplayProvider(store, a.run_id)
            res = AgentRuntime(rp, store=store).run(orig["task"])
            print(f"replay run_id: {res.run_id}\nanswer: {res.answer}")
            print("divergences:", rp.divergences or "none")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
