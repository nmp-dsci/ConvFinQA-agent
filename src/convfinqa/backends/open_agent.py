"""The open runtime (s25): the champion's single-session contract on an open model.

`agent_sdk` answers a conversation in one Claude Agent SDK session. This module
answers it in one Pydantic AI agent over one message history, on a DeepSeek
model through the DeepSeek API — so that the harness, the provider and the
weights are all open, and a trace holds every reasoning token.

Lift and shift means what the model is given is **imported, never copied**:

- the system prompt is the `sdk_vN` text the caller passes, as the agent's only
  instructions — no harness prompt is added;
- the turn messages are `agent_sdk._first_message` / `_later_message`;
- the six calculator tools keep the `mcp__cfq__*` names the prompt itself uses,
  the SDK's descriptions and its exact JSON schema, and run the same functions;
- the reply is `SdkTurnResult`, delivered through an output tool named
  `StructuredOutput`, which is how Claude Code delivers an `output_format`;
- a reply that still fails validation gets `_correction()`'s text, within the
  same `ATTEMPTS_PER_TURN`, and the capture is `result_to_capture`'s, so the
  scorer cannot tell the runtimes apart.

What differs is the harness and the model, which is the experiment, plus two
mechanics the harness forces:

- **The failed attempt leaves the history.** Claude's session keeps a rejected
  reply and answers the correction after it. Here a rejected run can end on a
  tool call with no tool result, which DeepSeek refuses, so the correction is
  sent with the question again on the history as it stood before the attempt.
- **A failed turn stays in the history.** Claude's session keeps the exchange
  of a turn that never produced a valid reply. Here that turn enters the
  history as its exact prompt and a plain-text reply saying no valid answer
  was produced — not the rejected attempts, which may end on an unpaired tool
  call — so the report document and every question stay in view.
- **Tool schemas are closed.** Pydantic AI's OpenAI schema transformer adds
  `"additionalProperties": false` to each tool's parameters; otherwise the
  schema is the SDK's, byte for byte.
- **Thinking is DeepSeek's protocol.** `reasoning_content` goes back on every
  tool-call turn and no tool is forced while thinking; both are Pydantic AI's
  DeepSeek profile (`llm.open_agent_model`), not code here.

The harness `pydantic_ai_checked` (s29) adds three turn checks for the SFT
student, H1–H3 in `open_checks`; the plain `pydantic_ai` harness is unchanged,
so llm_react_v1 and the data it collects are what they were.

A provider that will not answer — no balance, rate-limited or overloaded after
the transport's retries — raises `SdkRateLimitError`, exactly as a spent Claude
subscription does: the turn is unscored, never wrong, and the pass is finished
with `--resume-from`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from convfinqa.backends.agent_sdk import (
    _TOOL_FNS,
    _TOOL_SCHEMA,
    ATTEMPTS_PER_TURN,
    SdkRateLimitError,
    SdkTurnError,
    SdkTurnResult,
    _correction,
    _first_message,
    _later_message,
    _tokens_used,
    result_to_capture,
)
from convfinqa.data.schemas import ConversationHistory
from convfinqa.llm import SDK_CALCULATOR_TOOLS, SDK_MCP_SERVER
from convfinqa.tracking import tracing

RUNTIME = "open_agent"

# Claude Code's name for the tool an `output_format` is delivered through.
OUTPUT_TOOL_NAME = "StructuredOutput"

# The model-side errors that mean "no answer", not "a wrong answer": the
# provider is out of balance, rate-limiting, overloaded or unreachable after the
# transport's own retries. Named by class so a 400 — a request this code got
# wrong — is never mistaken for one.
_UNAVAILABLE = (
    "ModelQuotaExceededError",
    "ModelRateLimitError",
    "ModelUnavailableError",
    "ModelServerError",
    "ModelConnectionError",
    "ModelRequestTimeout",
    "LLMUnavailableError",
)
# The same by status, for a response the framework surfaces as a plain HTTP
# error: 402 is DeepSeek's "insufficient balance", 429 its rate limit, 503 its
# "server overloaded".
_UNAVAILABLE_STATUS = {402, 429, 500, 502, 503, 504}


def tool_description(name: str) -> str:
    """The description the SDK gives a calculator tool, verbatim."""
    return f"Return {name}(a, b). {_TOOL_FNS[name].__doc__}"


def _tool_function(
    name: str, trajectory: list[dict[str, Any]], *, checks: bool = False
) -> Callable[..., Awaitable[str]]:
    """One calculator, recording the same two events `agent_sdk` records.

    Under `checks` (H1) an argument beyond ``a`` and ``b`` is a tool error;
    without, it raises as it always did, so the plain harness is unchanged.
    """
    fn = _TOOL_FNS[name]

    async def call(a: Any = None, b: Any = None) -> str:
        trajectory.append(
            {"event": "tool_call", "tool": name, "args": {"a": a, "b": b}}
        )
        try:
            text = str(fn(float(a), float(b)))
        except (ZeroDivisionError, TypeError, ValueError, OverflowError) as exc:
            # A tool error is a message to the model, as in the SDK runtime.
            text = f"error: {exc}"
        trajectory.append({"event": "tool_return", "tool": name, "result": text})
        return text

    if not checks:
        return call

    async def checked(a: Any = None, b: Any = None, **extra: Any) -> str:
        if extra:
            from convfinqa.backends.open_checks import extra_argument_error

            args = {"a": a, "b": b, **extra}
            text = extra_argument_error(name, extra)
            trajectory.append({"event": "tool_call", "tool": name, "args": args})
            trajectory.append({"event": "tool_return", "tool": name, "result": text})
            return text
        return await call(a, b)

    return checked


def build_agent(
    system_prompt: str,
    model: Any,
    trajectory: list[dict[str, Any]],
    *,
    checks: dict[str, Any] | None = None,
) -> Any:
    """The open agent: the prompt, the six tools and the output tool, nothing else.

    `checks`, when given, is the checked harness's per-question state
    (``{"retried": False}``, reset by the caller at each question, then the
    name of the check that sent the reply back): the tools
    take H1 and the reply must pass H2 and H3, with one retry per question.
    Its output tool allows one more retry than the plain harness, so a failed
    check never spends the retry a malformed reply would need.
    """
    from pydantic_ai import Agent, ModelRetry, RunContext, Tool
    from pydantic_ai.output import ToolOutput

    tools = [
        Tool.from_schema(
            _tool_function(name, trajectory, checks=checks is not None),
            name=f"mcp__{SDK_MCP_SERVER}__{name}",
            description=tool_description(name),
            json_schema=dict(_TOOL_SCHEMA),
        )
        for name in SDK_CALCULATOR_TOOLS
    ]
    agent = Agent(
        model,
        instructions=system_prompt,
        tools=tools,
        output_type=ToolOutput(
            SdkTurnResult,
            name=OUTPUT_TOOL_NAME,
            max_retries=1 if checks is None else 2,
        ),
    )
    if checks is not None:
        from convfinqa.backends.open_checks import failed_check

        @agent.output_validator
        def _turn_checks(ctx: RunContext[Any], output: SdkTurnResult) -> SdkTurnResult:
            if checks["retried"]:
                return output
            failed = failed_check(output, list(ctx.messages))
            if failed is None:
                return output
            checks["retried"] = failed[0]
            raise ModelRetry(failed[1])

    return agent


def unavailable(exc: BaseException) -> str | None:
    """Why the provider gave no answer, or None when `exc` is a real failure."""
    seen: BaseException | None = exc
    while seen is not None:
        status = getattr(seen, "status_code", None)
        if type(seen).__name__ in _UNAVAILABLE or status in _UNAVAILABLE_STATUS:
            return f"{type(seen).__name__}: {seen}"
        seen = seen.__cause__ or seen.__context__
    return None


def _usage_of(messages: list[Any], model: str, at: datetime) -> dict[str, Any]:
    """The usage of every model response in `messages`, in the SDK's keys.

    `input_tokens` is the *new* input, cache reads excluded — the quantity
    `agent_sdk._tokens_used` bounds, so the ceiling counts what it counts for
    the champion. DeepSeek reports its cache hits as `prompt_cache_hit_tokens`
    where an OpenAI-shaped response says `cached_tokens`; both are read.
    """
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    from convfinqa.tracking import cost

    total_in = cache_read = output = reasoning = requests = 0
    thinking: list[str] = []
    served: set[str] = set()
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        requests += 1
        usage = message.usage
        details = dict(usage.details or {})
        hit = int(usage.cache_read_tokens or details.get("prompt_cache_hit_tokens", 0))
        total_in += int(usage.input_tokens or 0)
        cache_read += hit
        output += int(usage.output_tokens or 0)
        reasoning += int(
            usage.output_reasoning_tokens or details.get("reasoning_tokens", 0)
        )
        if message.model_name:
            served.add(str(message.model_name))
        thinking += [p.content for p in message.parts if isinstance(p, ThinkingPart)]
    new_input = max(0, total_in - cache_read)
    return {
        "num_turns": requests,
        "input_tokens": new_input,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 0,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_cost_usd": round(
            cost.deepseek_cost_usd(
                model,
                cache_hit_tokens=cache_read,
                cache_miss_tokens=new_input,
                output_tokens=output,
                at=at,
            ),
            8,
        ),
        "served_models": sorted(served),
        "thinking": thinking,
    }


NO_VALID_ANSWER = "I could not produce a valid answer to that question."


def _failed_turn(prompt: str) -> list[Any]:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    return [
        ModelRequest(parts=[UserPromptPart(content=prompt)]),
        ModelResponse(parts=[TextPart(content=NO_VALID_ANSWER)]),
    ]


async def _ask(
    agent: Any,
    prompt: str,
    history: list[Any],
    *,
    model: str,
    max_turns: int,
) -> tuple[SdkTurnResult, dict[str, Any], list[Any]]:
    """One question, with the champion's correction budget.

    Returns the validated reply, the turn's usage (every attempt's tokens
    counted, since the ceiling bounds what was spent) and the history to
    carry into the next question.
    """
    from pydantic_ai import capture_run_messages
    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
    from pydantic_ai.usage import UsageLimits

    started = time.perf_counter()
    spent: list[Any] = []
    message = prompt
    last = ""
    for attempt in range(ATTEMPTS_PER_TURN):
        with capture_run_messages() as attempted:
            try:
                result = await agent.run(
                    message,
                    message_history=list(history),
                    usage_limits=UsageLimits(request_limit=max_turns),
                )
            except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
                spent += attempted[len(history) :]
                last = str(exc)
                # The question again, with the correction after it: the
                # rejected attempt is not kept (see the module docstring).
                message = f"{prompt}\n\n{_correction(last)}"
                continue
            except Exception as exc:
                reason = unavailable(exc)
                if reason is not None:
                    raise SdkRateLimitError(reason) from exc
                raise SdkTurnError(f"{type(exc).__name__}: {exc}") from exc
        new = result.new_messages()
        usage = _usage_of(spent + new, model, datetime.now(UTC))
        usage["attempts"] = attempt + 1
        usage["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        # Thinking belongs to the reply that was accepted, not to rejected ones.
        usage["thinking"] = _usage_of(new, model, datetime.now(UTC))["thinking"]
        return result.output, usage, result.all_messages()
    raise SdkTurnError(f"no valid reply after {ATTEMPTS_PER_TURN} attempts: {last!r}")


async def run_conversation(
    report_id: str,
    questions: list[str],
    *,
    system_prompt: str,
    model: str,
    effort: str,
    captures: list[dict[str, Any]] | None = None,
    stop_after: Callable[[int, str], bool] | None = None,
    max_turns: int | None = None,
    version: str = "",
    trajectories: dict[str, Any] | None = None,
    harness: str = "pydantic_ai",
    temperature: float | None = None,
) -> tuple[list[str], list[str]]:
    """Walk one conversation over one message history; return (preds, programs).

    The contract of `agent_sdk.run_conversation`, turn for turn: a failed turn
    is recorded with ``capture["error"]`` and an empty answer that still
    enters the history; the token ceiling ends the walk the same way; a
    provider that will not answer raises `SdkRateLimitError` carrying the
    answers captured before it. `trajectories`, when given, receives the
    conversation's full message history under `report_id` — reasoning, tool
    calls and replies, losslessly — for the run's trajectory artifact.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    from convfinqa.backends.open_checks import CHECKED_HARNESS
    from convfinqa.config import settings
    from convfinqa.data.loader import _DOCS
    from convfinqa.llm import open_agent_model

    trajectory: list[dict[str, Any]] = []
    checks: dict[str, Any] | None = (
        {"retried": False} if harness == CHECKED_HARNESS else None
    )
    agent = build_agent(
        system_prompt,
        open_agent_model(model, effort, temperature=temperature),
        trajectory,
        checks=checks,
    )
    turns = max_turns or settings.sdk_max_turns
    document = _DOCS[report_id]
    conversation = ConversationHistory()
    history: list[Any] = []
    preds: list[str] = []
    programs: list[str] = []
    tokens_total = 0
    limit = settings.sdk_total_tokens_limit

    try:
        for i, question in enumerate(questions):
            hist_text = conversation.as_text()
            cap: dict[str, Any] = {"history_text": hist_text}
            answer, program = "", ""
            with tracing.span(
                f"q{i}: {question[:60]}",
                attributes={
                    "report_id": report_id,
                    "turn_index": i,
                    "question": question,
                    "runtime": RUNTIME,
                    "prompt_version": version,
                },
            ) as qspan:
                if tokens_total > limit:
                    cap["error"] = (
                        f"token budget exhausted: {tokens_total} > {limit} "
                        f"before q{i}; turn not attempted"
                    )
                else:
                    trajectory.clear()
                    if checks is not None:
                        checks.clear()
                        checks["retried"] = False
                    prompt = (
                        _first_message(report_id, document, question)
                        if i == 0
                        else _later_message(question)
                    )
                    try:
                        result, usage, history = await _ask(
                            agent, prompt, history, model=model, max_turns=turns
                        )
                    except SdkRateLimitError as exc:
                        cap["error"] = f"rate_limited: {exc.refusal}"
                        cap["rate_limited"] = True
                        qspan.set(rate_limited=True, error=exc.refusal)
                        if captures is not None:
                            captures.append(cap)
                        raise SdkRateLimitError(
                            exc.refusal, preds=preds, programs=programs, turn_index=i
                        ) from exc
                    except SdkTurnError as exc:
                        cap["error"] = str(exc)
                        history = [*history, *_failed_turn(prompt)]
                    else:
                        tokens_total += _tokens_used(usage)
                        cap.update(
                            result_to_capture(
                                result,
                                question=question,
                                history_text=hist_text,
                                trajectory=trajectory,
                                metrics=usage,
                            )
                        )
                        cap["sdk"]["served_models"] = usage["served_models"]
                        cap["sdk"]["reasoning_tokens"] = usage["reasoning_tokens"]
                        if checks is not None:
                            # which check sent this question back, if one did
                            cap["sdk"]["harness_retry"] = checks["retried"] or ""
                        answer = result.answer
                        program = (
                            result.program if result.turn_type == "program" else ""
                        )
                qspan.set(
                    answer=answer, program=program or None, tokens_total=tokens_total
                )
            preds.append(answer)
            programs.append(program)
            conversation.append(question=question, answer=answer, report_id=report_id)
            if captures is not None:
                captures.append(cap)
            if stop_after is not None and stop_after(i, answer):
                break
    finally:
        if trajectories is not None:
            trajectories[report_id] = ModelMessagesTypeAdapter.dump_python(
                history, mode="json"
            )
    return preds, programs
