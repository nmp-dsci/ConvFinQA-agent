"""Abuse controls and the keyless-boot property."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from convfinqa.serving.limits import (
    InFlightLimiter,
    RateLimited,
    SlidingWindowRateLimiter,
    TooManyInFlight,
    require_owner,
)


class _Client:
    def __init__(self, host: str) -> None:
        self.host = host


class _Request:
    """Minimal stand-in for a Starlette request."""

    def __init__(
        self, headers: dict[str, str] | None = None, client: _Client | None = None
    ) -> None:
        self.headers = headers or {}
        self.client = client


@pytest.mark.asyncio
async def test_inflight_cap_rejects_rather_than_queues() -> None:
    """Being told the demo is busy beats waiting 90 seconds behind three turns."""
    limiter = InFlightLimiter(2)
    await limiter.acquire()
    await limiter.acquire()
    with pytest.raises(TooManyInFlight):
        await limiter.acquire()
    await limiter.release()
    await limiter.acquire()
    assert limiter.active == 2


@pytest.mark.asyncio
async def test_inflight_release_never_goes_negative() -> None:
    """An extra release must not create phantom capacity."""
    limiter = InFlightLimiter(1)
    await limiter.release()
    await limiter.release()
    assert limiter.active == 0


def test_rate_limit_window_slides() -> None:
    """Requests expire out of the window rather than counting forever."""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
    limiter.check("ip", now=0.0)
    limiter.check("ip", now=1.0)
    with pytest.raises(RateLimited):
        limiter.check("ip", now=2.0)
    # Once the first two age out, the client is allowed again.
    limiter.check("ip", now=12.0)


def test_rate_limit_is_per_client() -> None:
    """One noisy visitor must not throttle everyone else."""
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
    limiter.check("a", now=0.0)
    limiter.check("b", now=0.0)
    with pytest.raises(RateLimited):
        limiter.check("a", now=1.0)


def test_rate_limit_table_is_pruned() -> None:
    """Without pruning the per-IP dict leaks one entry per visitor, forever."""
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=10)
    for i in range(50):
        limiter.check(f"ip-{i}", now=0.0)
    assert limiter.prune(now=100.0) == 50
    assert limiter._hits == {}


def test_owner_token_unset_means_refused_not_open() -> None:
    """A deployment that forgot to configure a token must not publish its writes."""
    from convfinqa.config import settings

    original = settings.owner_token
    settings.owner_token = None
    try:
        with pytest.raises(HTTPException) as exc:
            require_owner(_Request())  # type: ignore[arg-type]
        assert exc.value.status_code == 403
        assert exc.value.detail["code"] == "owner_token_unset"
    finally:
        settings.owner_token = original


def test_owner_token_must_match() -> None:
    """A wrong or absent token is refused; the right one passes."""
    from pydantic import SecretStr

    from convfinqa.config import settings

    original = settings.owner_token
    settings.owner_token = SecretStr("s3cret")
    try:
        with pytest.raises(HTTPException):
            require_owner(_Request({"x-owner-token": "wrong"}))  # type: ignore[arg-type]
        with pytest.raises(HTTPException):
            require_owner(_Request())  # type: ignore[arg-type]
        require_owner(_Request({"x-owner-token": "s3cret"}))  # type: ignore[arg-type]
    finally:
        settings.owner_token = original


def test_client_key_prefers_forwarded_for() -> None:
    """Behind App Runner, `request.client` is the balancer, not the visitor."""
    from convfinqa.serving.limits import client_key

    request = _Request({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
    assert client_key(request) == "203.0.113.9"  # type: ignore[arg-type]


def test_client_key_ignores_forwarded_for_when_proxy_is_untrusted() -> None:
    """With `trusted_proxy=False`, a caller cannot spoof the rate-limit key by
    sending its own X-Forwarded-For header — the socket peer is used instead."""
    from convfinqa.config import settings
    from convfinqa.serving.limits import client_key

    original = settings.trusted_proxy
    settings.trusted_proxy = False
    try:
        request = _Request(
            {"x-forwarded-for": "203.0.113.9, 10.0.0.1"},
            client=_Client("198.51.100.7"),
        )
        assert client_key(request) == "198.51.100.7"  # type: ignore[arg-type]
    finally:
        settings.trusted_proxy = original


def test_settings_boot_without_a_key() -> None:
    """A fresh clone and the keyless demo container must import cleanly."""
    from convfinqa.config import Settings

    settings = Settings(deepseek_api_key=None)
    assert settings.deepseek_api_key is None
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is not set"):
        settings.require_deepseek_api_key()
