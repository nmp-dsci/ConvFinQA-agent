"""Abuse controls for the public deployment: concurrency cap, per-IP rate limit.

Ordered cheapest-check-first, because the point is to shed load before spending
anything on it: a global in-flight semaphore rejects instantly, the per-IP window
costs one dict lookup, and only then does a request reach a handler.

In-memory state is *correct* here rather than a shortcut — App Runner runs this
service at max-size 1, so there is exactly one process and no shared store to be
consistent with. If that ever changes, these two classes are the seam.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from convfinqa.config import settings


class TooManyInFlight(HTTPException):
    """Raised when the global in-flight turn cap is already saturated."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "demo_full",
                "message": (
                    "The demo is handling as many conversations as it can right "
                    "now. Try again in a moment."
                ),
            },
        )


class RateLimited(HTTPException):
    """Raised when a client exceeds its sliding-window request allowance."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "demo_rate_limited",
                "message": f"Too many requests. Try again in {retry_after}s.",
            },
            headers={"Retry-After": str(retry_after)},
        )


class InFlightLimiter:
    """A non-blocking global cap on concurrent turns.

    Deliberately *not* a queue: a visitor who waits 90 seconds behind three other
    turns has a worse experience than one told immediately that the demo is busy.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        """Number of turns currently executing."""
        return self._active

    async def acquire(self) -> None:
        """Claim a slot, or raise `TooManyInFlight` if none is free."""
        async with self._lock:
            if self._active >= self.limit:
                raise TooManyInFlight()
            self._active += 1

    async def release(self) -> None:
        """Return a slot to the pool."""
        async with self._lock:
            self._active = max(0, self._active - 1)


class SlidingWindowRateLimiter:
    """Per-client request allowance over a sliding time window."""

    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1.0, window_seconds)
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def check(self, client_key: str, *, now: float | None = None) -> None:
        """Record a request from `client_key`, raising `RateLimited` if over quota."""
        moment = now if now is not None else time.monotonic()
        window = self._hits[client_key]
        cutoff = moment - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.max_requests:
            retry_after = max(1, int(window[0] + self.window_seconds - moment) + 1)
            raise RateLimited(retry_after)
        window.append(moment)

    def prune(self, *, now: float | None = None) -> int:
        """Drop clients with no recent activity. Returns how many were dropped.

        Without this the dict grows one entry per distinct IP, forever — a slow
        leak that only shows up in a long-lived process, which is exactly what
        this is.
        """
        moment = now if now is not None else time.monotonic()
        cutoff = moment - self.window_seconds
        stale = [
            key for key, hits in self._hits.items() if not hits or hits[-1] < cutoff
        ]
        for key in stale:
            del self._hits[key]
        return len(stale)


def client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    App Runner terminates TLS and forwards the caller in `X-Forwarded-For`, so
    the first hop in that list is the real client; `request.client` would report
    the load balancer and rate-limit every visitor as one.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_owner(request: Request) -> None:
    """Gate admin writes on the owner token.

    An unset token means *refused*, not *open* — a deployment that forgot to
    configure it must not thereby publish its promote and research endpoints.
    """
    expected = settings.owner_token
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "owner_token_unset",
                "message": "Admin writes are disabled: no OWNER_TOKEN is configured.",
            },
        )
    presented = request.headers.get("x-owner-token", "")
    if not presented or presented != expected.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "owner_token_invalid",
                "message": "This action requires the owner token.",
            },
        )
