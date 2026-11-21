"""Tool registry: schema inferred from type hints, safe execution with timeout + error capture."""
from __future__ import annotations

import inspect
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class ToolResult:
    ok: bool
    output: str
    error: str = ""

    def render(self) -> str:
        return self.output if self.ok else f"ERROR: {self.error}"


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict = field(default_factory=dict)
    timeout_s: float = 10.0

    def spec(self) -> str:
        props = ", ".join(
            f"{k}: {v['type']}" + ("" if k in self.parameters.get("required", []) else "?")
            for k, v in self.parameters.get("properties", {}).items()
        )
        return f"- {self.name}({props}): {self.description}"

    def run(self, args: dict) -> ToolResult:
        missing = [k for k in self.parameters.get("required", []) if k not in args]
        if missing:
            return ToolResult(False, "", f"missing arguments: {missing}")
        unknown = [k for k in args if k not in self.parameters.get("properties", {})]
        if unknown:
            return ToolResult(False, "", f"unknown arguments: {unknown}")
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["v"] = self.fn(**args)
            except Exception as e:  # tool errors are observations, not crashes
                box["e"] = f"{type(e).__name__}: {e}"

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(self.timeout_s)
        if th.is_alive():
            return ToolResult(False, "", f"timeout after {self.timeout_s}s")
        if "e" in box:
            return ToolResult(False, "", box["e"])
        v = box["v"]
        out = v if isinstance(v, str) else json.dumps(v, default=str)
        return ToolResult(True, out[:4000])  # cap observation size to protect the context window


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, fn: Callable | None = None, *, name: str | None = None, timeout_s: float = 10.0):
        def deco(f: Callable) -> Callable:
            hints = get_type_hints(f)
            sig = inspect.signature(f)
            props, required = {}, []
            for p in sig.parameters.values():
                props[p.name] = {"type": _JSON_TYPES.get(hints.get(p.name, str), "string")}
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
            desc = (inspect.getdoc(f) or "").split("\n")[0]
            tool = Tool(name or f.__name__, desc, f, {"properties": props, "required": required}, timeout_s)
            self._tools[tool.name] = tool
            return f

        return deco(fn) if fn else deco

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values()) or "(no tools)"

    def subset(self, names: list[str]) -> "ToolRegistry":
        r = ToolRegistry()
        for n in names:
            if n in self._tools:
                r.add(self._tools[n])
        return r
