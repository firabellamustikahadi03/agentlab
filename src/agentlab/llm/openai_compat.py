"""OpenAI-compatible chat provider (Groq, Gemini, OpenRouter, Ollama ...). stdlib only."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import LLMResponse, Message, ProviderError, Usage, estimate_tokens

PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
        "model": "llama3.2",
    },
}


class OpenAICompatProvider:
    def __init__(
        self,
        preset: str = "groq",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        price_per_mtok: tuple[float, float] = (0.0, 0.0),
        max_retries: int = 4,
        timeout: float = 60.0,
    ):
        cfg = PRESETS.get(preset, {})
        self.name = preset
        self.model = model or cfg.get("model", "")
        self.base_url = (base_url or cfg.get("base_url", "")).rstrip("/")
        key_env = cfg.get("key_env", "")
        self.api_key = api_key or (os.environ.get(key_env, "") if key_env else "")
        if preset != "ollama" and not self.api_key:
            raise ProviderError(f"Missing API key: set {key_env or 'api_key'}")
        self.price = price_per_mtok  # (input, output) USD per 1M tokens; free tiers = 0
        self.max_retries = max_retries
        self.timeout = timeout

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        start = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(2**attempt, 20))  # exponential backoff for free-tier limits
                    continue
                raise ProviderError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(min(2**attempt, 20))
        else:
            raise ProviderError(f"provider failed after retries: {last_err}")

        content = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        usage = Usage(
            u.get("prompt_tokens") or sum(estimate_tokens(m.content) for m in messages),
            u.get("completion_tokens") or estimate_tokens(content),
        )
        cost = (usage.prompt_tokens * self.price[0] + usage.completion_tokens * self.price[1]) / 1e6
        return LLMResponse(
            content=content,
            usage=usage,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.model,
            cost_usd=cost,
        )
