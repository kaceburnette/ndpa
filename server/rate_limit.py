"""
Simple in-memory rate limiter (token bucket per key).

This is per-process, so multi-instance deploys need Redis. For v1 with
a single Fly.io machine, this is fine. Swap to Redis if you go multi-region.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window_sec: int) -> None:
        """Raise 429 if over limit."""
        now = time.monotonic()
        cutoff = now - window_sec
        bucket = self._buckets[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded: {limit} requests / {window_sec}s",
            )
        bucket.append(now)
