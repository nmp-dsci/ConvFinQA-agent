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


# --- Claude Agent SDK (the single-session pipeline challenger) --------------

# The qa_agent runtime: one Claude session per conversation doing the whole
# triage → preprocess → retrieve → calculate job, with the six calculator
# functions as its only tools. A different model from the teacher on purpose —
# the teacher judges a few dozen cases per cycle; this runs on every turn of an
# eval pass, so its cost has to stay measurable per question. `settings.sdk_model`
# overrides the constant so a run can be pinned to a different model without a
# code change, and the run records which one it used.
LM_SDK_MODEL = "claude-sonnet-5"

# The one MCP server the runtime registers, and the only tools it may call.
# Restricting `tools`/`allowed_tools` to exactly these six is what makes
# "arithmetic happens in tools, not in the model's head" an enforced property
# rather than a prompt request — the trajectory the calculator stage records is
# then a complete account of every number the session computed.
SDK_MCP_SERVER = "cfq"
SDK_CALCULATOR_TOOLS = ("add", "subtract", "multiply", "divide", "exp", "greater")
SDK_ALLOWED_TOOLS = [f"mcp__{SDK_MCP_SERVER}__{name}" for name in SDK_CALCULATOR_TOOLS]


def sdk_model_name() -> str:
    """The model the qa_agent runtime runs on: the setting, else the constant."""
    return settings.sdk_model or LM_SDK_MODEL


def sdk_model_slug(model: str | None = None) -> str:
    """A short, filename-safe name for a model, for run names and CSV names.

    `claude-sonnet-5` → `sonnet-5`, `claude-haiku-4-5-20251001` → `haiku-4-5`.
    The run name is the one place a reader meets the model before opening the
    run, so it carries the family and version and drops the vendor prefix and
    the date snapshot. Runs recorded before this existed have no slug in their
    name; their model is the `sdk_model` param, which was always logged.
    """
    name = (model or sdk_model_name()).strip().lower()
    for prefix in ("claude-", "anthropic/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    parts = name.split("-")
    # Drop a trailing YYYYMMDD snapshot: it pins the weights, not the family.
    if len(parts) > 1 and len(parts[-1]) == 8 and parts[-1].isdigit():
        parts = parts[:-1]
    slug = "-".join(p for p in parts if p)
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in slug) or "model"


def api_env() -> dict[str, str]:
    """Environment for an Agent SDK child process billed to `ANTHROPIC_API_KEY`.

    The mirror image of `subscription_env`, and built here for the same reason:
    the child's environment decides who pays, nothing in the output says which
    path was taken, and the only evidence is the bill. So the key is *required*
    rather than hoped for — a missing key raises here, at the one place the
    environment is built, instead of surfacing as the CLI silently falling back
    to whatever login it finds in the keychain and billing the subscription for
    an eval pass the operator meant to measure per token.

    Everything else `subscription_env` strips is stripped here too. `CLAUDECODE`
    and the `CLAUDE_CODE_*` session variables would make the child inherit the
    identity of the Claude Code session the loop is driven from; blanking
    `CLAUDE_CODE_OAUTH_TOKEN` stops a dotfile token from turning an API-billed
    run into a subscription one halfway through.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set, and SDK_BILLING=api needs it: the "
            "Agent SDK child would otherwise fall back to the CLI's own login "
            "and bill the subscription for a run meant to be measured per token. "
            "Set the key in ~/.env or the process environment, or select "
            "SDK_BILLING=subscription deliberately."
        )
    drop = {"CLAUDECODE"}
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in drop and not k.startswith("CLAUDE_CODE_")
    }
    env["ANTHROPIC_API_KEY"] = key
    env["CLAUDE_CODE_OAUTH_TOKEN"] = ""
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


# The qa_agent runtime runs on the subscription, and only on the subscription
# (owner's instruction, 2026-09-05). The API path stays implemented — it is the
# mirror image that makes `api_env`'s guarantees testable, and a replacement
# runtime that one day ships would need it — but it may not be *selected* for a
# run of this loop. Two reasons, one of each kind:
#
#   * Money. An eval pass is 349 questions and a campaign is five of them plus
#     diagnosis and rewrite traffic. The subscription covers that as time; the
#     API covers it as an open-ended bill nobody approved per pass.
#   * Evidence. The key in this environment answers `Credit balance is too low`,
#     and the Agent SDK returns that as the *reply text* rather than an error, so
#     the first pass on the API path scored 176 refusals as wrong answers and
#     reported 44.4% as though it had measured something. The failure is silent
#     at the call site; the gate has to be shut here instead.
#
# `SDK_ALLOW_API_BILLING=1` is the deliberate escape hatch, so the refusal is a
# decision to reverse rather than a wall to work around.
SDK_BILLING_SUBSCRIPTION = "subscription"
SDK_BILLING_API = "api"
API_BILLING_ESCAPE_HATCH = "SDK_ALLOW_API_BILLING"


def api_billing_allowed() -> bool:
    """True only when the operator has explicitly re-opened the API path."""
    return os.environ.get(API_BILLING_ESCAPE_HATCH, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def sdk_child_env(billing: str | None = None) -> dict[str, str]:
    """The child environment for the qa_agent runtime, by billing path.

    `billing` defaults to `settings.sdk_billing`, which is `subscription`. The
    API path is refused unless `SDK_ALLOW_API_BILLING=1` — see the note above:
    an API-billed pass is both unbudgeted and, with an exhausted key, silently
    unmeasurable, and this is the one place the choice is made.
    """
    chosen = billing or settings.sdk_billing
    if chosen == SDK_BILLING_API:
        if not api_billing_allowed():
            raise RuntimeError(
                "SDK_BILLING=api is refused: the qa_agent runtime runs on the "
                "subscription (owner's instruction, 2026-09-05). An API-billed "
                "pass is unbudgeted, and an exhausted key returns 'Credit "
                "balance is too low' as a reply rather than an error, which "
                "scores as 176 wrong answers instead of a failure. Use "
                f"SDK_BILLING=subscription, or set {API_BILLING_ESCAPE_HATCH}=1 "
                "to re-open the path deliberately."
            )
        return api_env()
    if chosen == SDK_BILLING_SUBSCRIPTION:
        return subscription_env()
    raise ValueError(
        f"unknown SDK billing path {chosen!r}: use 'api' or 'subscription'"
    )


def pipeline_sdk_options(
    *,
    system_prompt: str,
    mcp_server: Any,
    allowed_tools: list[str],
    output_schema: dict[str, Any],
    max_turns: int,
    billing: str | None = None,
    model: str | None = None,
) -> Any:
    """Options for one qa_agent session. Sibling of `teacher_options`.

    Same gate, same `setting_sources=[]` — the runtime must not inherit this
    repository's CLAUDE.md or skills, because it is the thing being evaluated
    and an inherited instruction file would silently become part of its prompt.

    What differs from the teacher: `tools=[]` disables every built-in tool (no
    Bash, no file reads — the document arrives in the first user message and
    that is all the session may consult), `mcp_servers` carries the in-process
    calculator server, and `allowed_tools` is exactly the six calculator names
    so nothing needs a permission prompt and nothing else can run. The
    structured `output_format` is what turns the session's reply into the
    per-turn capture the shared scorer reads.

    `allowed_tools` is checked against `SDK_ALLOWED_TOOLS` rather than trusted:
    a caller that widened it would widen what the model can do without any
    trace of it in the run's params.
    """
    guard_llm_call()
    from claude_agent_sdk import ClaudeAgentOptions

    if set(allowed_tools) != set(SDK_ALLOWED_TOOLS):
        raise ValueError(
            f"the qa_agent runtime may call exactly {SDK_ALLOWED_TOOLS}; "
            f"got {allowed_tools}"
        )
    return ClaudeAgentOptions(
        model=model or sdk_model_name(),
        system_prompt=system_prompt,
        env=sdk_child_env(billing),
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        setting_sources=[],
        tools=[],
        allowed_tools=list(SDK_ALLOWED_TOOLS),
        mcp_servers={SDK_MCP_SERVER: mcp_server},
        output_format={"type": "json_schema", "schema": output_schema},
    )
