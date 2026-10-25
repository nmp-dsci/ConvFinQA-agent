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
from typing import Any, Literal, TypeVar

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


def teacher_model_name() -> str:
    """The model the teacher runs on: `settings.teacher_model`, else the constant.

    The judge loop (s12) pins the diagnose/distil teacher to Sonnet 5 rather
    than the loop's default Opus, and every teacher run records which one it
    used (`actor_model`, `diagnoser_model`), so the override is a setting and
    not an edit to the constant.
    """
    return settings.teacher_model or LM_TEACHER_MODEL


# The confidence judge (s12): a second, cheaper model that reads a finished
# turn's trace and says whether to release the answer. It runs on every served
# turn, so it is the cheapest model that can do the job — and it is calibrated
# to one answering model, so the run that scores it records both.
LM_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def judge_model_name() -> str:
    """The model the confidence judge runs on: the setting, else the constant."""
    return settings.judge_model or LM_JUDGE_MODEL


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
    """Drop the cached providers. For tests that flip settings between cases."""
    global _provider, _open_provider, _student_provider
    _provider = None
    _open_provider = None
    _student_provider = None


# --- The open runtime (s25): sdk_v1's contract on an open model ---------------
#
# The same DeepSeek account as the pipeline, and a different model call: the
# pipeline switches thinking off (`DISABLE_THINKING_BODY`), while the open
# agent is the champion's contract lifted onto an open model and so reasons, at
# an effort that is part of what the agent is. DeepSeek's thinking protocol —
# send `reasoning_content` back on every tool-call turn, never force a tool
# while thinking — is Pydantic AI's DeepSeek profile, which is why it is used.
OPEN_HARNESSES: tuple[str, ...] = ("pydantic_ai",)
DEEPSEEK_OPEN_MODELS: tuple[str, ...] = ("deepseek-flash", "deepseek-v4-pro")
# The SFT student (s28): Qwen3-4B-Instruct-2507 on vLLM on Modal, untrained
# and with the sft1 adapter. A non-thinking model, so its effort is "none":
# its reasoning is the visible `<reasoning>` text it was trained to write.
STUDENT_MODELS: tuple[str, ...] = ("student-qwen3-4b", "student-qwen3-4b-sft1")
OPEN_MODELS: tuple[str, ...] = (*DEEPSEEK_OPEN_MODELS, *STUDENT_MODELS)
OPEN_EFFORTS: tuple[str, ...] = ("low", "high", "max", "none")
OPEN_BILLING = "deepseek-api"
STUDENT_BILLING = "modal"
# Qwen's recommended sampling for Instruct-2507, as tau-telecom-sft served it.
STUDENT_SAMPLING: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 4096,
    "extra_body": {"top_k": 20},
}

# `deepseek-flash` is DeepSeek V4.1 Flash, but Pydantic AI recognises a V4 model
# only by the `deepseek-v4-` prefix (pydantic-ai #8353). Profiled under its own
# name it would be treated as a model that cannot think, and forced into a tool
# call that thinking mode rejects with a 400; so it is profiled as the V4 model
# it is. `deepseek-v4-flash` is the retired name DeepSeek routes to it.
_OPEN_PROFILE_AS = {"deepseek-flash": "deepseek-v4-flash"}

_open_provider: Any = None


def _open_http_client() -> httpx.AsyncClient:
    """The open runtime's HTTP client: the pipeline's timeout and retry policy."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
        transport=_RetryTransport(
            max_attempts=settings.llm_max_attempts,
            timeout=settings.llm_timeout_seconds,
        ),
    )


def open_agent_model(model: str, effort: str) -> OpenAIChatModel:
    """A DeepSeek chat model for the open runtime, thinking at `effort`.

    Refuses a model or effort outside the declared sets rather than passing it
    through: DeepSeek maps unknown efforts silently (`xhigh` becomes `high`),
    and a run named for one effort that ran at another is not that agent's
    evidence.
    """
    guard_llm_call()
    check_open_agent(model, effort)
    if model in STUDENT_MODELS:
        return _student_model(model)
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    global _open_provider
    if _open_provider is None:
        _open_provider = DeepSeekProvider(
            api_key=settings.require_deepseek_api_key(),
            http_client=_open_http_client(),
        )
    return OpenAIChatModel(
        model,
        provider=_open_provider,
        profile=DeepSeekProvider.model_profile(_OPEN_PROFILE_AS.get(model, model)),
        settings=ModelSettings(thinking=effort),  # type: ignore[typeddict-item]
    )


def check_open_agent(model: str, effort: str) -> None:
    """Refuse a model, effort or pairing outside the declared sets.

    DeepSeek thinks at low/high/max; a student does not think at all, so it
    takes "none" and DeepSeek never does.
    """
    if model not in OPEN_MODELS:
        raise ValueError(f"unknown open model {model!r}; expected one of {OPEN_MODELS}")
    if effort not in OPEN_EFFORTS:
        raise ValueError(
            f"unknown open effort {effort!r}; expected one of {OPEN_EFFORTS}"
        )
    if (model in STUDENT_MODELS) != (effort == "none"):
        raise ValueError(
            f"open effort {effort!r} does not fit {model!r}: a student takes 'none', "
            "a DeepSeek model one of low/high/max"
        )


_student_provider: Any = None


def _student_model(model: str) -> OpenAIChatModel:
    """The student on vLLM, with tool choice left to the model.

    Pydantic AI forces a tool call when the only output is a tool, and vLLM
    honours a forced call with guided decoding, which would leave no room for
    the `<reasoning>` text the student was trained to write before each call.
    So the profile declares no forced tool choice (`tool_choice="auto"`),
    exactly as DeepSeek's thinking profile does.
    """
    global _student_provider
    if _student_provider is None:
        if not settings.student_url:
            raise RuntimeError(
                "STUDENT_URL is not set: deploy the student server "
                "(`convfinqa-sft serve`) and put its URL in ~/.env"
            )
        headers = {}
        if settings.modal_proxy_token_id and settings.modal_proxy_token_secret:
            headers = {
                "Modal-Key": settings.modal_proxy_token_id.get_secret_value(),
                "Modal-Secret": settings.modal_proxy_token_secret.get_secret_value(),
            }
        _student_provider = OpenAIProvider(
            base_url=f"{settings.student_url.rstrip('/')}/v1",
            api_key="modal-proxy",  # Modal's proxy authenticates; vLLM takes no key
            http_client=httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
                transport=_RetryTransport(
                    max_attempts=settings.llm_max_attempts,
                    timeout=settings.llm_timeout_seconds,
                ),
            ),
        )
    return OpenAIChatModel(
        model,
        provider=_student_provider,
        profile=_student_profile(),
        settings=ModelSettings(**STUDENT_SAMPLING),  # type: ignore[typeddict-item]
    )


def _student_profile() -> Any:
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    return OpenAIModelProfile(
        supports_forced_tool_choice=False,
        openai_supports_strict_tool_definition=False,
    )


def student_wire_model(
    transport: httpx.AsyncBaseTransport, model: str = STUDENT_MODELS[0]
) -> OpenAIChatModel:
    """The student's model over `transport` in place of the server.

    The SFT builder sends one request through a capturing transport to read
    the exact system message and tool JSON the student is served, so training
    rows carry those bytes rather than a reconstruction. No network, no key.
    """
    provider = OpenAIProvider(
        base_url="http://student.invalid/v1",
        api_key="capture",
        http_client=httpx.AsyncClient(transport=transport),
    )
    return OpenAIChatModel(
        model,
        provider=provider,
        profile=_student_profile(),
        settings=ModelSettings(**STUDENT_SAMPLING),  # type: ignore[typeddict-item]
    )


def open_model_slug(harness: str, model: str) -> str:
    """The run-name slug of an open agent: `pyd-deepseek-flash`."""
    short = {"pydantic_ai": "pyd"}.get(harness, harness.replace("_", "-"))
    return f"{short}-{model}"


#: What the app calls an open agent: a harness running a prompt. The prompt is
#: sdk_v1 byte for byte, so the agent cannot take the prompt's name: "sdk-v1"
#: on a DeepSeek pass reads as the champion. The model and effort stay beside
#: the name, as they do for a Claude agent. Run names, CSVs and ledger rows
#: keep what they recorded; this is the label the pages show.
OPEN_AGENT_NAMES: dict[tuple[str, str], str] = {
    ("pydantic_ai", "sdk_v1"): "llm_react_v1",
}


def open_agent_name(harness: str, version: str) -> str | None:
    """`llm_react_v1` for Pydantic AI running sdk_v1; None for an unnamed pair."""
    return OPEN_AGENT_NAMES.get((harness, version))


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
    except TimeoutError as exc:
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
    model: str | None = None,
) -> Any:
    """Options for one Agent SDK call. The demo gate applies here as everywhere.

    `model` defaults to the teacher's; the judge passes its own so the one
    chokepoint (`evalloop/sdk.py::run_structured`) serves both.

    Note `setting_sources=[]`: the teacher must not inherit this repository's
    CLAUDE.md, settings or skills. It is being asked to judge a pipeline, not to
    behave like a contributor to the project, and an inherited instruction file
    would silently become part of its prompt.

    `effort="low"`: left unset, the SDK defaults every call here to `"high"`
    (2026-09-10 decision) — every Agent SDK call in this project, chosen
    explicitly rather than inherited from the SDK's own default.
    """
    guard_llm_call()
    from claude_agent_sdk import ClaudeAgentOptions

    kwargs: dict[str, Any] = {
        "model": model or teacher_model_name(),
        "system_prompt": system_prompt,
        "env": subscription_env(),
        "max_turns": max_turns,
        "permission_mode": "bypassPermissions",
        "setting_sources": [],
        "allowed_tools": allowed_tools or [],
        "effort": "low",
    }
    if output_schema is not None:
        kwargs["output_format"] = {"type": "json_schema", "schema": output_schema}
    return ClaudeAgentOptions(**kwargs)


# --- Claude Agent SDK (the single-session pipeline challenger) --------------

# The qa_agent runtime: one Claude session per conversation doing the whole
# triage → preprocess → retrieve → calculate job, with the six calculator
# functions as its only tools. A different model from the teacher on purpose —
# the teacher judges a few dozen cases per cycle; this runs on every turn of an
# eval pass, so its cost has to stay measurable per question.
#
# The model and effort normally come from the champion record
# (`registry.champion_agent()`): the champion is an agent, and serving and eval
# passes run it as recorded. `settings.sdk_model` / `settings.sdk_effort`
# override that for a process, and a pass can pin either; every run records
# which it used. The constants are the fallback for a registry with no record.
LM_SDK_MODEL = "claude-sonnet-5"
SdkEffort = Literal["low", "medium", "high", "xhigh", "max"]
LM_SDK_EFFORT: SdkEffort = "low"
SDK_EFFORTS: tuple[SdkEffort, ...] = ("low", "medium", "high", "xhigh", "max")
# Commit 27e11b5 pinned `effort="low"` on every Agent SDK call. A run started
# before it left effort unset, and `ClaudeAgentOptions` then defaults to
# "high" — so the committed sdk_v1 evidence (2026-09-05/06) was measured at
# high. Readers of a run with no `sdk_effort` param date it against this.
SDK_EFFORT_PINNED_LOW_AT = "2026-09-10T04:04:46+00:00"

# The one MCP server the runtime registers, and the only tools it may call.
# Restricting `tools`/`allowed_tools` to exactly these six is what makes
# "arithmetic happens in tools, not in the model's head" an enforced property
# rather than a prompt request — the trajectory the calculator stage records is
# then a complete account of every number the session computed.
SDK_MCP_SERVER = "cfq"
SDK_CALCULATOR_TOOLS = ("add", "subtract", "multiply", "divide", "exp", "greater")
SDK_ALLOWED_TOOLS = [f"mcp__{SDK_MCP_SERVER}__{name}" for name in SDK_CALCULATOR_TOOLS]


def _champion_agent() -> dict[str, Any]:
    """The champion record, or {} when the registry has none or cannot be read."""
    try:
        from convfinqa.tracking import registry

        return registry.champion_agent() or {}
    except Exception:  # noqa: BLE001 — a missing registry falls back to the constants
        return {}


def sdk_model_name() -> str:
    """The model the qa_agent runtime runs on: the setting, else the champion's, else the constant."""
    return settings.sdk_model or _champion_agent().get("model") or LM_SDK_MODEL


def sdk_effort_name() -> str:
    """The effort the qa_agent runtime runs at: the setting, else the champion's, else low."""
    effort = settings.sdk_effort or _champion_agent().get("effort") or LM_SDK_EFFORT
    if effort not in SDK_EFFORTS:
        raise ValueError(f"unknown effort {effort!r}; expected one of {SDK_EFFORTS}")
    return str(effort)


def sdk_effort_of_run(params: dict[str, Any], start_time_ms: float | None) -> str:
    """The effort a recorded sdk run ran at.

    New runs carry it as the `sdk_effort` param. Older ones do not, and their
    effort depends on when they ran: before `SDK_EFFORT_PINNED_LOW_AT` the SDK
    default (`high`), after it the pinned `low`. A run with no start time is
    read as pinned, the conservative answer for anything recent.
    """
    recorded = params.get("sdk_effort")
    if recorded:
        return str(recorded)
    if start_time_ms is None:
        return LM_SDK_EFFORT
    from datetime import datetime

    pinned = datetime.fromisoformat(SDK_EFFORT_PINNED_LOW_AT).timestamp() * 1000
    return "high" if float(start_time_ms) < pinned else LM_SDK_EFFORT


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
    effort: str | None = None,
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

    `effort` defaults to `sdk_effort_name()` — the champion's recorded effort.
    The teacher and judge keep their own `low` in `teacher_options`; only the
    answering agent's effort is part of what the champion is.
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
        effort=_checked_effort(effort),
    )


def _checked_effort(effort: str | None) -> SdkEffort:
    """`effort`, or the resolved default, refused when it is not a known level."""
    resolved = effort or sdk_effort_name()
    for level in SDK_EFFORTS:
        if resolved == level:
            return level
    raise ValueError(f"unknown effort {resolved!r}; expected one of {SDK_EFFORTS}")
