"""Provider-agnostic LLM types.

Agents talk to models through a plain JSON protocol (not vendor tool-calling),
so any chat model, including small free ones, can drive the runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    cost_usd: float = 0.0


class Provider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse: ...


class ProviderError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used when a provider reports no usage."""
    return max(1, len(text) // 4)
