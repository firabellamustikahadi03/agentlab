"""Built-in eval suites. Add your own by returning a list[EvalTask]."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalTask:
    id: str
    prompt: str
    checks: list[dict]
    tags: list[str] = field(default_factory=list)


def _py(code: str) -> str:
    return f"Run this code and report exactly what it prints:\n```python\n{code}\n```"


BASIC = [
    EvalTask("math-1", "What is 17*23+5?", [{"type": "numeric", "value": 396}], ["math"]),
    EvalTask("math-2", "What is (144/12)+81/9?", [{"type": "numeric", "value": 21}], ["math"]),
    EvalTask("math-3", "What is 2^10 - 24?", [{"type": "numeric", "value": 1000}], ["math"]),
    EvalTask("math-4", "What is 7*8*9?", [{"type": "numeric", "value": 504}], ["math"]),
    EvalTask(
        "tool-1", "Use the calculator: what is 999*999?",
        [{"type": "numeric", "value": 998001}, {"type": "tool_used", "tool": "calculator"}], ["tools"],
    ),
    EvalTask(
        "code-1", _py("print(sum(range(1, 101)))"),
        [{"type": "numeric", "value": 5050}, {"type": "tool_used", "tool": "python"}], ["code", "tools"],
    ),
    EvalTask("code-2", _py("print(len(set('mississippi')))"), [{"type": "numeric", "value": 4}], ["code"]),
    EvalTask(
        "code-3", _py("a, b = 0, 1\nfor _ in range(20):\n    a, b = b, a + b\nprint(a)"),
        [{"type": "numeric", "value": 6765}], ["code"],
    ),
]

REASONING = [
    EvalTask(
        "word-1",
        "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. What is the total distance in km?",
        [{"type": "numeric", "value": 270}], ["reasoning"],
    ),
    EvalTask(
        "word-2",
        "A shop sells pens at 3 for $4.50. How much do 14 pens cost in dollars?",
        [{"type": "numeric", "value": 21}], ["reasoning"],
    ),
    EvalTask(
        "word-3",
        "I have 5 boxes with 12 apples each. I give away a third of all apples, then eat 4. How many are left?",
        [{"type": "numeric", "value": 36}], ["reasoning"],
    ),
    EvalTask(
        "word-4",
        "What is the sum of all prime numbers below 50? Use code if helpful.",
        [{"type": "numeric", "value": 328}], ["reasoning", "code"],
    ),
]

SUITES = {"basic": BASIC, "reasoning": REASONING, "all": BASIC + REASONING}


def get_suite(name: str) -> list[EvalTask]:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; choose from {sorted(SUITES)}")
    return SUITES[name]
