"""The single-session runtime: one Claude Agent SDK session per conversation.

The four-agent pipeline (`pipeline/runner.py`) runs triage, preprocess, retrieve
and calculate as four separate model calls per turn, each with its own prompt.
This runtime does the same job in **one** Claude session per conversation: the
report arrives in the first user message, every later question is one more
message in the same session, and the model's reply is a structured object that
carries what each of the four stages would have produced. The six calculator
functions are its only tools, so every number it computes goes through a tool
call and lands in the trajectory the calculator stage records.

Everything after the capture dict is shared. `result_to_capture` writes the
exact shape `pipeline/runner.py::turn_events` fills, so `_capture_to_row_fields`,
`stage_scores.score_rows` and `stage_scores.first_fault` read an sdk run without
knowing which runtime produced it — that is what makes an A/B on the eval split
a comparison of runtimes rather than of two scoring paths.

Three things the pipeline gets for free and this runtime has to do by hand:

- **Tracing.** The SDK spawns the `claude` CLI as a subprocess, so no autologger
  sees it. Each SDK turn opens an `LLM` span in the pattern of
  `evalloop/sdk.py::run_structured` (one per attempt), each tool call a `TOOL`
  span, and prompts are stored by reference (`prompt_refs.sdk_prompt_ref`).
- **History text.** The model sees the session; the scorer and the teacher
  read `history_text`, so a local `ConversationHistory` is kept and rendered
  exactly as the pipeline renders it — including the runtime's own wrong
  answers, which is the conversation contract.
- **Budget.** Tokens are summed across the session and the remaining turns are
  recorded as error rows once `settings.sdk_total_tokens_limit` is exceeded.
- **Refusals.** The `claude` CLI answers a spent account with prose — "You've
  hit your session limit · resets 5:40pm", "Credit balance is too low" — which
  is not a wrong answer but *no answer*. `SdkRateLimitError` separates the two:
  it is never retried (a second ask buys the same refusal) and it aborts the
  conversation, so the runner can write those turns as unscored instead of
  scoring 176 of 349 refusals as failures, as one live pass did.

`claude_agent_sdk` is imported lazily and `ClaudeSDKClient` may be constructed
here and in `evalloop/sdk.py` only — a test pins both.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from convfinqa.data.schemas import ConversationHistory, ConvType, TurnType
from convfinqa.llm import SDK_ALLOWED_TOOLS, SDK_CALCULATOR_TOOLS, SDK_MCP_SERVER
from convfinqa.pipeline import tools as calc_tools
from convfinqa.pipeline.wire_format import render_chat_inputs
from convfinqa.tracking import tracing

RUNTIME = "agent_sdk"

# How many times one question is asked before the turn is recorded as an
# error: the first ask plus one corrective message when the reply does not
# validate. A second correction has never been seen to help and costs a turn of
# a budget shared by the rest of the conversation.
ATTEMPTS_PER_TURN = 2

# A short head of the user prompt goes on the span; the rest is a reference.
TRACED_HEAD_CHARS = 400

# The CLI answers a spent account with prose where an object should be. Each
# entry is a *conjunction* of lowercase substrings, so "session limit" only
# counts as a refusal when "resets" is there too — a model may legitimately
# discuss a limit, and a financial answer may legitimately say "quota", but
# neither arrives as unstructured prose from a session that had a schema.
# The refusal vocabulary and its classifier live in `evalloop/sdk.py`, the other
# sanctioned SDK call site and the lower layer (this module already imports its
# `TeacherCallError`). Shared rather than copied: both chokepoints must agree on
# what a refusal looks like, and a marker added in one place must not be missing
# from the other. Re-exported here so this module stays the qa_agent's one door.
from convfinqa.evalloop.sdk import (  # noqa: E402 — re-export, kept next to its users
    RATE_LIMIT_ERROR_PREFIX,
    RATE_LIMIT_MARKERS,
    rate_limit_refusal,
)

__all__ = ["RATE_LIMIT_ERROR_PREFIX", "RATE_LIMIT_MARKERS", "rate_limit_refusal"]


class SdkRuntimeUnavailableError(RuntimeError):
    """`claude_agent_sdk` is not importable in this environment."""


class SdkTurnError(RuntimeError):
    """One question got no reply that validates as `SdkTurnResult`."""


class SdkRateLimitError(SdkTurnError):
    """The CLI refused the turn: session limit, rate limit, or no credit.

    Not a wrong answer — *no* answer, and the difference is the whole point of
    the class. It is raised in place of `SdkTurnError` so the corrective retry
    is skipped (a second ask buys the same refusal) and so `run_conversation`
    can abort: the session is spent, and every later turn of the same
    conversation would refuse identically.

    When it leaves `run_conversation` it carries the answers captured *before*
    the refusal, which is how the runner tells a turn that was answered from
    one that was never attempted.
    """

    def __init__(
        self,
        refusal: str,
        *,
        preds: list[str] | None = None,
        programs: list[str] | None = None,
        turn_index: int | None = None,
    ) -> None:
        super().__init__(refusal)
        self.refusal = refusal
        self.preds: list[str] = list(preds or [])
        self.programs: list[str] = list(programs or [])
        self.turn_index = turn_index


def _load_sdk() -> Any:
    """The `claude_agent_sdk` module, imported on first use.

    Lazy so that importing this module needs neither the package nor a key —
    the demo container imports everything and must not fail on a runtime it
    never runs.
    """
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError as exc:  # pragma: no cover — environment, not a code path
        raise SdkRuntimeUnavailableError(
            "the agent_sdk runtime needs the claude-agent-sdk package and the "
            "`claude` CLI on PATH"
        ) from exc


# --- The structured output contract ------------------------------------------


def _as_str(value: Any) -> Any:
    """Numbers arrive as numbers from a JSON-schema reply; the CSV wants text.

    `str(150.0)` is `"150.0"`, which is what the pipeline's calculator writes
    for the same value (`str(part.content)` of a float tool return), so the two
    runtimes spell an answer the same way.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    return value


class RetrievedItem(BaseModel):
    """One sub-question and the value the session found for it."""

    question: str
    answer: str
    # Where the value came from — a table cell, a sentence — kept for the
    # teacher; the pipeline's retriever has no equivalent field.
    source: str = ""

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: Any) -> Any:
        return _as_str(value)


class SdkTurnResult(BaseModel):
    """What one question's reply must contain: the four stages' outputs at once.

    Field names follow the pipeline's stage models (`TriageOut`,
    `PreprocessOut`, `RetrievedValues`, `CalcOut`) so the mapping to a capture
    is a rename, not a translation. `program` is symbolic (`divide(A, B)`,
    placeholders bound to `sub_questions` in order) exactly as preprocess
    emits it, because `stage_scores` binds and executes it the same way.
    """

    turn_type: TurnType
    conv_type: ConvType
    sub_questions: list[str] = Field(default_factory=list)
    program: str = ""
    retrieved: list[RetrievedItem] = Field(default_factory=list)
    answer: str
    reasoning: str = ""

    @field_validator("answer", "program", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        return _as_str(value)


# --- Tools: the six calculator functions as an in-process MCP server ----------

_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}

_TOOL_FNS: dict[str, Callable[[float, float], Any]] = {
    "add": calc_tools.add,
    "subtract": calc_tools.subtract,
    "multiply": calc_tools.multiply,
    "divide": calc_tools.divide,
    "exp": calc_tools.exp,
    "greater": calc_tools.greater,
}


def _text(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def _tool_handler(
    name: str, fn: Callable[[float, float], Any], trajectory: list[dict[str, Any]]
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        a, b = args.get("a"), args.get("b")
        # The same two events, in the same shape, that
        # `pipeline/runner.py::_calc_trajectory` replays from pydantic-ai.
        trajectory.append(
            {"event": "tool_call", "tool": name, "args": {"a": a, "b": b}}
        )
        with tracing.span(name, span_type="TOOL") as span:
            span.inputs({"a": a, "b": b})
            try:
                result = fn(float(a), float(b))  # type: ignore[arg-type]
            except (ZeroDivisionError, TypeError, ValueError, OverflowError) as exc:
                # A tool error is a message to the model, never an exception —
                # raising here would end the session, not the calculation.
                text = f"error: {exc}"
                trajectory.append(
                    {"event": "tool_return", "tool": name, "result": text}
                )
                span.set(error=text)
                return _text(text, is_error=True)
            text = str(result)
            trajectory.append({"event": "tool_return", "tool": name, "result": text})
            span.outputs(text)
            return _text(text)

    return handler


def build_calculator_server(sdk: Any, trajectory: list[dict[str, Any]]) -> Any:
    """The `cfq` MCP server: six calculator tools writing into `trajectory`.

    The handlers close over the list rather than returning results to a caller
    because the SDK owns the loop: the only way a tool call reaches the capture
    is by recording itself. The caller clears the list between turns.

    Registered by *calling* `sdk.tool(...)` rather than decorating, so this
    module keeps concrete annotations under a dynamically imported SDK.
    """
    tools = [
        sdk.tool(name, f"Return {name}(a, b). {_TOOL_FNS[name].__doc__}", _TOOL_SCHEMA)(
            _tool_handler(name, _TOOL_FNS[name], trajectory)
        )
        for name in SDK_CALCULATOR_TOOLS
    ]
    return sdk.create_sdk_mcp_server(name=SDK_MCP_SERVER, tools=tools)


# --- From a validated reply to the pipeline's capture dict --------------------

_STAGE_METRIC_KEYS = ("latency_ms", "input_tokens", "output_tokens")


def _stage_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """The subset of a turn's usage that a pipeline stage's `metrics` carries.

    One SDK call stands in for four stage calls, so every stage of a turn
    reports the same latency and tokens. Rolling them up per turn (as
    `tracking.usage` does) would then count the call four times, which is why
    the full accounting lives under ``capture["sdk"]`` instead.
    """
    return {k: metrics[k] for k in _STAGE_METRIC_KEYS if metrics.get(k) is not None}


def _inline_arithmetic(answer: str, trajectory: list[dict[str, Any]]) -> bool:
    """Did a program turn's answer come from somewhere other than a tool return?

    True when tools were called and the answer matches none of their returns:
    the session did some of the sum in its head. A turn with no tool call at
    all is a *skip* (recorded separately) rather than inline arithmetic — the
    two are different failures and counting one as both would double it.
    """
    from convfinqa.evaluation.metrics import numeric_match

    returns = [e["result"] for e in trajectory if e.get("event") == "tool_return"]
    if not returns:
        return False
    return not any(numeric_match(answer, r) for r in returns)


def result_to_capture(
    result: SdkTurnResult,
    *,
    question: str,
    history_text: str,
    trajectory: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Write a validated reply in the shape `turn_events` fills.

    The keys, the nesting and the None rules are the pipeline's: a number turn
    has `preprocess` and `calculator` set to None and the retriever holds one
    answer; a program turn has all four. `metrics` is the turn's usage
    (`latency_ms`, `num_turns`, `total_cost_usd`, token counts) and is split
    between the stages' `metrics` and ``capture["sdk"]``.
    """
    stage_metrics = _stage_metrics(metrics)
    triage_out = {
        "reasoning": result.reasoning,
        "turn_type": result.turn_type,
        "conv_type": result.conv_type,
    }
    capture: dict[str, Any] = {
        "history_text": history_text,
        "triage": {
            "input": {"question": question},
            "output": triage_out,
            "reasoning": result.reasoning,
            "metrics": stage_metrics,
        },
    }
    tool_calls = sum(1 for e in trajectory if e.get("event") == "tool_call")
    stage_skips: list[str] = []
    inline = False

    if result.turn_type == "number":
        capture["preprocess"] = None
        capture["retriever"] = {
            "input": {
                "turn_type": "number",
                "questions": [question],
                "history": history_text,
            },
            "output": {
                "reasoning": "",
                "answers": [{"question": question, "answer": result.answer}],
                "sources": [r.source for r in result.retrieved],
            },
            "reasoning": "",
            "metrics": stage_metrics,
        }
        capture["calculator"] = None
    else:
        answers = [
            {"question": r.question, "answer": r.answer} for r in result.retrieved
        ]
        capture["preprocess"] = {
            "input": {
                "question": question,
                "history": history_text,
                "conv_type": result.conv_type,
            },
            "output": {
                "reasoning": "",
                "sub_questions": list(result.sub_questions),
                "program": result.program,
            },
            "reasoning": "",
            "metrics": stage_metrics,
        }
        capture["retriever"] = {
            "input": {
                "turn_type": "program",
                "questions": list(result.sub_questions),
                "history": history_text,
            },
            "output": {
                "reasoning": "",
                "answers": answers,
                # Not a pipeline field; kept beside `answers` (never inside
                # them) so `_retrieved_values` reads the same list either way.
                "sources": [r.source for r in result.retrieved],
            },
            "reasoning": "",
            "metrics": stage_metrics,
        }
        capture["calculator"] = {
            "input": {
                "question": question,
                "retrieved": answers,
                "program": result.program,
            },
            "output": {"answer": result.answer},
            "trajectory": list(trajectory),
            "metrics": stage_metrics,
        }
        if not result.program.strip():
            stage_skips.append("preprocess")
        if not result.retrieved:
            stage_skips.append("retriever")
        if tool_calls == 0:
            stage_skips.append("calculator")
        inline = _inline_arithmetic(result.answer, trajectory)

    capture["sdk"] = {
        "num_turns": metrics.get("num_turns"),
        "tool_calls": tool_calls,
        "stage_skips": stage_skips,
        "inline_arithmetic": inline,
        "cost_usd": metrics.get("total_cost_usd"),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "cache_read_input_tokens": metrics.get("cache_read_input_tokens"),
    }
    return capture


# --- Driving the session -----------------------------------------------------


def _first_message(report_id: str, document: Any, question: str) -> str:
    """Turn 0's user message: the whole report, then the first question.

    Rendered with the same `render_chat_inputs` the pipeline hands its
    retriever, so the document reaches both runtimes in the same bytes.
    """
    return render_chat_inputs(
        {"report_id": report_id, "document": document, "question": question}
    )


def _later_message(question: str) -> str:
    return render_chat_inputs({"question": question})


def _correction(error: str) -> str:
    return (
        "Your last reply did not match the required output schema:\n"
        f"{error[:800]}\n"
        "Answer the same question again, replying only with a valid object."
    )


def _payload_of(structured: Any, texts: list[str]) -> Any:
    """The JSON object in a reply: the structured result, else parsed text.

    A reply that is prose rather than an object is checked for a refusal
    first — that is the shape a rate limit arrives in, and calling it a schema
    violation would spend a retry on it and then score it as a wrong answer.
    """
    from convfinqa.evalloop.sdk import TeacherCallError, _extract_json

    payload: Any = structured
    if not isinstance(payload, dict):
        refusal = rate_limit_refusal(payload, *texts)
        if refusal is not None:
            raise SdkRateLimitError(refusal)
    try:
        if isinstance(payload, str):
            payload = _extract_json(payload)
        if not isinstance(payload, dict):
            if not texts:
                raise SdkTurnError("the SDK returned no content at all")
            payload = _extract_json("\n".join(texts))
    except TeacherCallError as exc:
        raise SdkTurnError(str(exc)) from exc
    return payload


def _tokens_of(usage: dict[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        value = (usage or {}).get(key)
        if isinstance(value, int):
            out[key] = value
    return out


async def _ask_once(client: Any, sdk: Any, prompt: str) -> tuple[Any, dict[str, Any]]:
    """Send one message on the open session and collect its reply.

    Returns the parsed payload (not yet validated) and the turn's usage. The
    result message's `structured_output` is read first (`structured_result` is
    accepted for older SDKs), the text of the reply is the fallback.
    """
    texts: list[str] = []
    tools_used: list[str] = []
    usage: dict[str, Any] = {}
    structured: Any = None
    started = time.perf_counter()

    await client.query(prompt)
    async for message in client.receive_response():
        if isinstance(message, sdk.AssistantMessage):
            for block in message.content:
                if isinstance(block, sdk.TextBlock):
                    texts.append(block.text)
                elif isinstance(block, sdk.ToolUseBlock):
                    tools_used.append(block.name)
        elif isinstance(message, sdk.ResultMessage):
            usage = {
                "duration_ms": getattr(message, "duration_ms", None),
                "num_turns": getattr(message, "num_turns", None),
                "total_cost_usd": getattr(message, "total_cost_usd", None),
                **_tokens_of(getattr(message, "usage", None)),
            }
            structured = getattr(message, "structured_output", None)
            if structured is None:
                structured = getattr(message, "structured_result", None)
            if structured is None:
                structured = getattr(message, "result", None)
            if getattr(message, "is_error", False) and structured is None:
                errors = getattr(message, "errors", None) or []
                detail = (
                    f"{getattr(message, 'subtype', '')} "
                    f"{' '.join(str(e) for e in errors)}"
                ).strip()
                refusal = rate_limit_refusal(detail, *texts)
                if refusal is not None:
                    raise SdkRateLimitError(refusal)
                raise SdkTurnError(f"the SDK reported an error: {detail}")
    usage["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    usage["tools_used"] = tools_used
    return _payload_of(structured, texts), usage


async def _ask(
    client: Any,
    sdk: Any,
    prompt: str,
    *,
    refs: dict[str, Any],
    system_prompt: str,
    model: str,
    max_turns: int,
) -> tuple[SdkTurnResult, dict[str, Any]]:
    """Ask one question, validating the reply; one corrective retry.

    Opens an `LLM` span per attempt, as `run_structured` does, so a reply that
    failed to validate is visible as a failed call followed by a corrected one.

    A refusal (`SdkRateLimitError`) leaves immediately and is *not* retried:
    the correction would be answered by the same refusal, one call later.
    """
    from convfinqa.evalloop import prompt_refs

    last: Exception | None = None
    message = prompt
    for attempt in range(ATTEMPTS_PER_TURN):
        with tracing.span(
            "agent_sdk SdkTurnResult",
            span_type="LLM",
            attributes={
                "model": model,
                "schema": SdkTurnResult.__name__,
                "attempt": attempt + 1,
                "max_attempts": ATTEMPTS_PER_TURN,
                "max_turns": max_turns,
                "allowed_tools": list(SDK_ALLOWED_TOOLS),
            },
        ) as span:
            span.inputs(
                {
                    "refs": refs,
                    "system_prompt_sha": prompt_refs.sha(system_prompt),
                    "prompt_sha": prompt_refs.sha(message),
                    "prompt_chars": len(message),
                    "prompt_head": message[:TRACED_HEAD_CHARS],
                }
            )
            try:
                payload, usage = await _ask_once(client, sdk, message)
                parsed = SdkTurnResult.model_validate(payload)
            except SdkRateLimitError as exc:
                span.set(error=repr(exc), rate_limited=True)
                raise
            except (SdkTurnError, ValidationError) as exc:
                span.set(error=repr(exc))
                last = exc
                message = _correction(str(exc))
                continue
            span.outputs(parsed.model_dump())
            span.set(
                duration_ms=usage.get("duration_ms"),
                num_turns=usage.get("num_turns"),
                tools_used=usage.get("tools_used"),
                total_cost_usd=usage.get("total_cost_usd"),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            )
            return parsed, usage
    raise SdkTurnError(f"no valid reply after {ATTEMPTS_PER_TURN} attempts: {last!r}")


def _tokens_used(usage: dict[str, Any]) -> int:
    """The tokens a turn *added* — the quantity the ceiling bounds.

    Cache reads are excluded on purpose. A session re-reads its whole context
    (report, prior turns, tool calls) on every turn, and the SDK reports that as
    ``cache_read_input_tokens``: on the first smoke run 99.99% of the counted
    tokens were cache reads, and every conversation hit the 60k ceiling after
    its second turn. Those tokens are context already paid for, billed at a
    tenth of the input rate; what a looping model actually spends is new input,
    new output and cache writes, so that is what is counted.
    """
    return sum(
        int(usage.get(k) or 0)
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
        )
    )


async def run_conversation(
    report_id: str,
    questions: list[str],
    *,
    system_prompt: str,
    captures: list[dict[str, Any]] | None = None,
    stop_after: Callable[[int, str], bool] | None = None,
    model: str | None = None,
    billing: str | None = None,
    max_turns: int | None = None,
    version: str = "",
) -> tuple[list[str], list[str]]:
    """Walk one conversation in one SDK session; return (preds, programs).

    The contract of `ConversationRunner.run_conversation`: one prediction and
    one program per question attempted, a capture per question when `captures`
    is given, and `stop_after(i, answer)` ending the walk early. `version`
    names the `sdk_vN` module `system_prompt` came from, for the trace's
    prompt reference.

    A turn that fails — no valid reply, or the token ceiling reached — is
    recorded with ``capture["error"]`` and an empty answer, and that empty
    answer still enters the history: the later turns are asked in the
    conversation the runtime actually had, as they are in the pipeline.

    A *refusal* is different and ends the walk. The refused turn's capture is
    appended with ``error`` prefixed ``rate_limited: `` and
    ``rate_limited=True``, then `SdkRateLimitError` propagates carrying the
    answers captured before it: the session is spent, so the remaining turns
    were never attempted and the runner must not score them.
    """
    from convfinqa.config import settings
    from convfinqa.data.loader import _DOCS
    from convfinqa.evalloop import prompt_refs
    from convfinqa.llm import pipeline_sdk_options, sdk_model_name

    sdk = _load_sdk()
    trajectory: list[dict[str, Any]] = []
    server = build_calculator_server(sdk, trajectory)
    turns = max_turns or settings.sdk_max_turns
    model_name = model or sdk_model_name()
    options = pipeline_sdk_options(
        system_prompt=system_prompt,
        mcp_server=server,
        allowed_tools=list(SDK_ALLOWED_TOOLS),
        output_schema=SdkTurnResult.model_json_schema(),
        max_turns=turns,
        billing=billing,
        model=model_name,
    )
    refs = prompt_refs.sdk_prompt_ref(version, system_prompt) if version else {}
    document = _DOCS[report_id]
    conversation = ConversationHistory()
    preds: list[str] = []
    programs: list[str] = []
    tokens_total = 0
    limit = settings.sdk_total_tokens_limit

    async with sdk.ClaudeSDKClient(options=options) as client:
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
                },
            ) as qspan:
                if tokens_total > limit:
                    cap["error"] = (
                        f"token budget exhausted: {tokens_total} > {limit} "
                        f"before q{i}; turn not attempted"
                    )
                else:
                    trajectory.clear()
                    prompt = (
                        _first_message(report_id, document, question)
                        if i == 0
                        else _later_message(question)
                    )
                    try:
                        result, usage = await _ask(
                            client,
                            sdk,
                            prompt,
                            refs=refs,
                            system_prompt=system_prompt,
                            model=model_name,
                            max_turns=turns,
                        )
                    except SdkRateLimitError as exc:
                        cap["error"] = RATE_LIMIT_ERROR_PREFIX + exc.refusal
                        cap["rate_limited"] = True
                        qspan.set(rate_limited=True, error=exc.refusal)
                        if captures is not None:
                            captures.append(cap)
                        raise SdkRateLimitError(
                            exc.refusal,
                            preds=preds,
                            programs=programs,
                            turn_index=i,
                        ) from exc
                    except SdkTurnError as exc:
                        cap["error"] = str(exc)
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
    return preds, programs
