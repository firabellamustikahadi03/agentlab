from .base import LLMResponse, Message, Provider, ProviderError, Usage, estimate_tokens
from .mock import MockAgentProvider, ScriptedProvider
from .openai_compat import PRESETS, OpenAICompatProvider
from .throttle import RateLimiter, ThrottledProvider


def make_provider(spec: str):
    """Build a provider: mock, mock-flaky, mock-gullible, mock-brittle, or <preset>[:<model>] (groq, gemini:...)."""
    if spec == "mock":
        return MockAgentProvider()
    if spec == "mock-flaky":
        return MockAgentProvider(name="mock-flaky", fail_first=True)
    if spec == "mock-gullible":
        return MockAgentProvider(name="mock-gullible", gullible=True)
    if spec == "mock-brittle":
        return MockAgentProvider(name="mock-brittle", retries=False)
    preset, _, model = spec.partition(":")
    return OpenAICompatProvider(preset=preset, model=model or None)


__all__ = [
    "LLMResponse", "Message", "Provider", "ProviderError", "Usage", "estimate_tokens",
    "MockAgentProvider", "ScriptedProvider", "OpenAICompatProvider", "PRESETS", "make_provider",
    "RateLimiter", "ThrottledProvider",
]
