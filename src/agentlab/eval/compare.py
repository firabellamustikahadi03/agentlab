"""Diff two eval runs: per-config summary deltas and per-task pass-rate changes."""
from __future__ import annotations

from dataclasses import dataclass, field

from .harness import EvalReport

_METRICS = ("success_rate", "avg_tokens", "avg_latency_s", "avg_cost_usd", "avg_llm_calls")


@dataclass
class Comparison:
    a_id: str
    b_id: str
    configs: dict[str, dict] = field(default_factory=dict)  # config -> {metric: (a, b, delta)}
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)  # {task, config, a, b, change}

    def regressions(self, max_drop: float = 0.05) -> list[str]:
        """Configs whose success rate fell by more than `max_drop`, plus every task that went pass -> fail."""
        out = []
        for c, m in self.configs.items():
            a, b, d = m["success_rate"]
            if d < -max_drop:
                out.append(f"{c}: success {a:.0%} -> {b:.0%}")
        out += [
            f"{t['config']}/{t['task']}: {t['a']:.0%} -> {t['b']:.0%}"
            for t in self.tasks if t["change"] == "regressed"
        ]
        return out

    def to_markdown(self) -> str:
        lines = [f"# Eval comparison `{self.a_id}` -> `{self.b_id}`", "", "## Per config", ""]
        lines += [
            "| config | success | avg tokens | avg latency (s) | avg cost ($) | llm calls |",
            "|---|---|---|---|---|---|",
        ]
        for c, m in self.configs.items():
            sr, tk, lt, co, ll = (m[k] for k in _METRICS)
            lines.append(
                f"| {c} | {sr[0]:.0%} -> {sr[1]:.0%} ({sr[2] * 100:+.0f}pp) "
                f"| {tk[0]:.0f} -> {tk[1]:.0f} ({tk[2]:+.0f}) "
                f"| {lt[0]:.2f} -> {lt[1]:.2f} ({lt[2]:+.2f}) | {co[0]:.5f} -> {co[1]:.5f} ({co[2]:+.5f}) "
                f"| {ll[0]:.1f} -> {ll[1]:.1f} ({ll[2]:+.1f}) |"
            )
        for name, cs in (("only in A", self.only_in_a), ("only in B", self.only_in_b)):
            if cs:
                lines += ["", f"Configs {name}: " + ", ".join(cs)]
        changed = [t for t in self.tasks if t["change"] != "same"]
        lines += ["", "## Per task changes", ""]
        if not changed:
            lines.append("No per-task differences.")
        else:
            lines += ["| task | config | A | B | change |", "|---|---|---|---|---|"]
            lines += [
                f"| {t['task']} | {t['config']} | {t['a']:.0%} | {t['b']:.0%} | {t['change']} |" for t in changed
            ]
        return "\n".join(lines) + "\n"


def compare_reports(a: EvalReport, b: EvalReport) -> Comparison:
    sa, sb = a.summary(), b.summary()
    cmp = Comparison(a.eval_id, b.eval_id)
    cmp.only_in_a = [c for c in sa if c not in sb]
    cmp.only_in_b = [c for c in sb if c not in sa]
    for c in sa:
        if c in sb:
            cmp.configs[c] = {k: (sa[c][k], sb[c][k], sb[c][k] - sa[c][k]) for k in _METRICS}
    ta, tb = a.task_metrics(), b.task_metrics()
    for task in sorted(set(ta) | set(tb)):
        for c in cmp.configs:
            if c in ta.get(task, {}) and c in tb.get(task, {}):
                pa, pb = ta[task][c]["pass_rate"], tb[task][c]["pass_rate"]
                change = "regressed" if pb < pa else "improved" if pb > pa else "same"
                cmp.tasks.append({"task": task, "config": c, "a": pa, "b": pb, "change": change})
    return cmp
