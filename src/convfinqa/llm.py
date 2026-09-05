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
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
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

# The teacher and prompt writer run on Claude via the Agent SDK, on the owner's
# subscription — a different provider from the pipeline entirely, and
# deliberately so. The pipeline agents are what is being optimised and run on
# every turn, so their cost has to stay measurable per question; the teacher runs
# a few dozen times per cycle and its job is judgement, which is where the
# stronger model earns its keep.
LM_TEACHER_MODEL = "claude-opus-5"

# DeepSeek v4 turned thinking mode *on by default*, and a thinking-mode request
# rejects the `tool_choice` pydantic-ai sends for every structured `output_type`:
#
#     400 — "Thinking mode does not support this tool_choice"
#
# Every stage of the pipeline uses a structured output, so with thinking left at
# its default *every live turn fails on triage* — the first model call it makes.
# Disabling thinking at the one place models are built means no call site has to
# know this, and a stage added later inherits the fix rather than rediscovering
# the 400. The knob is provider-specific, hence `extra_body` rather than the
# typed `thinking` field, which pydantic-ai maps to a different wire shape.
DISABLE_THINKING_BODY: dict[str, Any] = {"thinking": {"type": "disabled"}}


def model_settings() -> ModelSettings:
    """Default per-model settings applied to every model this module builds."""
    return ModelSettings(extra_body=dict(DISABLE_THINKING_BODY))


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
    return OpenAIChatModel(
        model_name or LM_MINI_MODEL,
        provider=get_provider(),
        settings=model_settings(),
    )


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


# --- Claude Agent SDK (the teacher and prompt writer) -----------------------


def subscription_env() -> dict[str, str]:
    """Environment for an Agent SDK child process, with the API key removed.

    `ANTHROPIC_API_KEY` present in the child's environment makes the Claude CLI
    authenticate as an API client and bill per token, silently, even though the
    account has a subscription that would have covered the call. Nothing in the
    output says which path was taken — the only evidence is the bill. So the key
    is stripped here, at the one place the child environment is built, rather
    than trusted to be absent.

    Stripping the `CLAUDE_CODE_*` session variables matters for the same reason
    and was found the same way: when the loop is driven from inside a Claude Code
    session, the child inherits that session's identity and bills against it —
    the observed symptom was a bare "Credit balance is too low" from an account
    with an active subscription. A teacher call must stand on its own.
    """
    drop = {"ANTHROPIC_API_KEY", "CLAUDECODE"}
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in drop and not k.startswith("CLAUDE_CODE_")
    }
    # Omitting a variable is NOT enough: the SDK merges this mapping over the
    # parent's environment, so a key only disappears if it is explicitly blanked.
    # And `config.load_dotenv` puts the whole of ~/.env into os.environ at import
    # time (dspy reads DEEPSEEK_API_KEY from there), which is how a key nobody
    # exported reaches this process in the first place.
    env["ANTHROPIC_API_KEY"] = ""
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    # Only override the CLI's own stored login when a token is configured
    # explicitly. Injecting one unconditionally is worse than injecting none:
    # a stale token in a dotfile silently replaces a working keychain login,
    # and the failure it produces ("Credit balance is too low") names neither.
    if settings.teacher_oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = settings.teacher_oauth_token.get_secret_value()
    else:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = ""
    return env


def teacher_options(
    *,
    system_prompt: str,
    output_schema: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    max_turns: int = 12,
) -> Any:
    """Options for one Agent SDK call. The demo gate applies here as everywhere.

    Note `setting_sources=[]`: the teacher must not inherit this repository's
    CLAUDE.md, settings or skills. It is being asked to judge a pipeline, not to
    behave like a contributor to the project, and an inherited instruction file
    would silently become part of its prompt.
    """
    guard_llm_call()
    from claude_agent_sdk import ClaudeAgentOptions

    kwargs: dict[str, Any] = {
        "model": LM_TEACHER_MODEL,
        "system_prompt": system_prompt,
        "env": subscription_env(),
        "max_turns": max_turns,
        "permission_mode": "bypassPermissions",
        "setting_sources": [],
        "allowed_tools": allowed_tools or [],
    }
    if output_schema is not None:
        kwargs["output_format"] = {"type": "json_schema", "schema": output_schema}
    return ClaudeAgentOptions(**kwargs)
