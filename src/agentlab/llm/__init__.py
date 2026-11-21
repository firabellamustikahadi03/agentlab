from .base import LLMResponse, Message, Provider, ProviderError, Usage, estimate_tokens
from .mock import MockAgentProvider, ScriptedProvider
from .openai_compat import PRESETS, OpenAICompatProvider


def make_provider(spec: str):
    """`mock`, `mock-flaky`, or `<preset>[:<model>]` e.g. `groq`, `gemini:gemini-2.0-flash`."""
    if spec == "mock":
        return MockAgentProvider()
    if spec == "mock-flaky":
        return MockAgentProvider(name="mock-flaky", fail_first=True)
    preset, _, model = spec.partition(":")
    return OpenAICompatProvider(preset=preset, model=model or None)


__all__ = [
    "LLMResponse", "Message", "Provider", "ProviderError", "Usage", "estimate_tokens",
    "MockAgentProvider", "ScriptedProvider", "OpenAICompatProvider", "PRESETS", "make_provider",
]
