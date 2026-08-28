"""The single choke point every LLM call passes through.

Two things are true of this module and of no other:

1. **Nothing else constructs a model.** `backends.pydantic`, `backends.dspy`,
   and the s7 diagnosis agents all obtain their models here. That is what makes
   the demo gate real rather than advisory — a handler cannot route around a
   check it does not know exists, and `test_demo_mode.py` asserts the property
   by construction rather than by inspection.
2. **Nothing else decides retry policy.** A provider hiccup degrades to a clean,
   typed error instead of a hung request, and it degrades the same way on the
   serving path, the eval path, and the optimizer path.

The demo deployment holds no keys at all, so the gate has to fire *before* the
provider is constructed, not when the request fails.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from convfinqa.config import settings

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek model identifiers — the v4 names, after the 2026-07-24 deprecation of
# the legacy `deepseek-chat` / `deepseek-reasoner` aliases.
#   MINI = `deepseek-v4-flash` (284B/13B MoE): the four pipeline agents, every turn.
#   MAX  = `deepseek-v4-pro`  (1.6T/49B MoE): the s7 diagnostic router and the
#          four specialist Fix agents, where reasoning quality is the product.
LM_MINI_MODEL = "deepseek-v4-flash"

T = TypeVar("T")


class DemoModeError(RuntimeError):
    """Raised when a live model call is attempted while DEMO_MODE is set.

    Carries the stable error code the frontend maps to its own copy — handlers
    translate this to HTTP 501 with `code: not_available_demo`.
    """

    code = "not_available_demo"

    def __init__(self) -> None:
        super().__init__(
            "This deployment runs in demo mode and makes no model calls. "
            "Chat is served from a recorded pack; every read-only surface "
            "(reports, splits, answers, traces, experiments) is genuinely live."
        )


class LLMUnavailableError(RuntimeError):
    """Raised when the provider could not be reached after every retry."""

    code = "llm_unavailable"


def demo_mode_enabled() -> bool:
    """True when this process must not make model calls."""
    return bool(settings.demo_mode)


def guard_llm_call() -> None:
    """Refuse the call when running in demo mode. The gate, in one function.

    Called by every model constructor below, so a new call site inherits the
    gate by construction rather than by remembering to add it.
    """
    if demo_mode_enabled():
        raise DemoModeError()


# Errors worth retrying: transport failures and the provider's own 429/5xx.
# A 400 (bad request) or 401 (bad key) is not transient and must surface at once
# rather than being retried four times into a timeout.
_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    LLMUnavailableError,
)


def _should_retry_status(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


class _RetryTransport(httpx.AsyncHTTPTransport):
    """Retry transient provider failures with exponential backoff and jitter.

    Retrying at the transport layer rather than around each agent call means the
    policy applies uniformly to pydantic-ai, DSPy, and any raw httpx caller,
    without any of them opting in.
    """

    def __init__(self, *, max_attempts: int, timeout: float) -> None:
        super().__init__(retries=0)
        self._max_attempts = max(1, max_attempts)
        self._timeout = timeout

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send `request`, retrying transient failures up to `max_attempts`."""
        last: BaseException | None = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await super().handle_async_request(request)
                except _RETRYABLE as exc:
                    last = exc
                    raise
                if _should_retry_status(
                    httpx.HTTPStatusError("", request=request, response=response)
                ):
                    # Drain before retrying, or the connection leaks.
                    await response.aread()
                    await response.aclose()
                    raise LLMUnavailableError(
                        f"provider returned {response.status_code}"
                    )
                return response
        raise LLMUnavailableError(str(last) if last else "provider unreachable")


_provider: OpenAIProvider | None = None


def get_provider() -> OpenAIProvider:
    """Return the shared DeepSeek provider, constructing it on first use.

    Lazy on purpose: constructing it eagerly at import would demand a key from
    every process that merely imports the package, which is exactly what the
    keyless clone and the demo container must not require.
    """
    guard_llm_call()
    global _provider
    if _provider is None:
        _provider = OpenAIProvider(
            base_url=DEEPSEEK_BASE_URL,
            api_key=settings.require_deepseek_api_key(),
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
                transport=_RetryTransport(
                    max_attempts=settings.llm_max_attempts,
                    timeout=settings.llm_timeout_seconds,
                ),
            ),
        )
    return _provider


def get_model(model_name: str | None = None) -> OpenAIChatModel:
    """Return a chat model for `model_name`, defaulting to the mini pipeline model."""
    return OpenAIChatModel(model_name or LM_MINI_MODEL, provider=get_provider())


def reset_provider() -> None:
    """Drop the cached provider. For tests that flip settings between cases."""
    global _provider
    _provider = None


async def call_with_budget(
    fn: Callable[[], Awaitable[T]],
    *,
    timeout: float | None = None,
) -> T:
    """Await `fn()` under a hard wall-clock ceiling.

    The transport retries a stalled *connection*; this bounds the whole
    operation, including a provider that accepts the request and then streams
    nothing. Without it a single wedged turn holds a session lock forever.
    """
    limit = timeout if timeout is not None else settings.llm_timeout_seconds
    try:
        return await asyncio.wait_for(fn(), timeout=limit)
    except asyncio.TimeoutError as exc:
        raise LLMUnavailableError(
            f"model call exceeded {limit:.0f}s and was abandoned"
        ) from exc


def dspy_lm_kwargs(model: str | None = None) -> dict[str, Any]:
    """Keyword arguments for `dspy.LM`, routed through the same gate and key.

    DSPy constructs its own client, so it cannot share the retry transport; what
    it does share is the demo gate and the single source of the key.
    """
    guard_llm_call()
    return {
        "model": f"openai/{model or LM_MINI_MODEL}",
        "api_key": settings.require_deepseek_api_key(),
        "api_base": DEEPSEEK_BASE_URL,
    }
