"""Built-in tools: safe calculator, sandboxed python, text utilities."""
from __future__ import annotations

import ast
import math
import operator as op
import os
import subprocess
import sys
import tempfile

from .registry import ToolRegistry

_BIN = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "sin": math.sin, "cos": math.cos,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic via the AST: no names, attributes or arbitrary calls."""
    def ev(n: ast.AST):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            lhs, rhs = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(rhs) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](lhs, rhs)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    return ev(ast.parse(expr.replace("^", "**"), mode="eval").body)


def _limits() -> None:  # runs in the child process before exec (POSIX only)
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64)) if hasattr(resource, "RLIMIT_NPROC") else None


def run_python(code: str, timeout_s: float = 8.0) -> str:
    """Run Python in an isolated subprocess (empty env, temp cwd, CPU/memory/file limits)."""
    with tempfile.TemporaryDirectory() as d:
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=d, env={}, capture_output=True, text=True, timeout=timeout_s,
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"python exceeded {timeout_s}s") from None
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "").strip().splitlines()[-1] if p.stderr else f"exit {p.returncode}")
    return out or "(no output)"


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '(12+8)*3/4'. Supports sqrt, abs, round, min, max, log, pi."""
        v = safe_eval(expression)
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(round(v, 10))

    @reg.register(name="python", timeout_s=12.0)
    def python(code: str) -> str:
        """Run a short Python snippet in a sandbox and return stdout. Use print() for results."""
        return run_python(code)

    @reg.register
    def word_count(text: str) -> str:
        """Count words and characters in a text."""
        return f"words={len(text.split())} chars={len(text)}"

    return reg
