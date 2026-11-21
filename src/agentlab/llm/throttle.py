"""Client-side rate limiting for free-tier APIs: a thread-safe limiter and a provider wrapper."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .base import LLMResponse, Message, Provider


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart, across any number of threads.

    Each caller reserves the next free slot under the lock and sleeps OUTSIDE it, so waiting
    threads queue up in order instead of serialising on the lock. `clock`/`sleep` are injectable
    so tests need no real time.
    """

    def __init__(
        self,
        rpm: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.interval = 60.0 / rpm
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> float:
        """Block until a slot is free; return how long this caller waited (seconds)."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return max(wait, 0.0)


class ThrottledProvider:
    """Wraps a provider so every LLM call goes through a RateLimiter (name/model are forwarded)."""

    def __init__(self, inner: Provider, limiter: RateLimiter):
        self.inner, self.limiter = inner, limiter
        self.name, self.model = inner.name, inner.model

    def complete(self, messages: list[Message], temperature: float = 0.0) -> LLMResponse:
        self.limiter.acquire()
        return self.inner.complete(messages, temperature)
