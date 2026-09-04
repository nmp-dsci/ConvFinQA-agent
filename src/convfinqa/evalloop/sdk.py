"""The Agent SDK half of the teacher: one call, one validated object, its usage.

The teacher and the prompt writer run on Claude through the Agent SDK on the
owner's subscription, which buys three things the previous pydantic-ai
implementation could not have:

- **Tools.** The writer can read its own history out of MLflow rather than being
  handed a fixed excerpt of it. What it actually read is visible in the message
  stream and recorded on the run.
- **A stronger judge.** Attribution and prompt surgery are the two places in
  this system where reasoning quality is the product.
- **Subscription billing**, provided ``ANTHROPIC_API_KEY`` is not in the child's
  environment — see ``llm.subscription_env``, which is the only place that
  environment is built.

What it does *not* change is the contract: every call still returns an instance
of the same pydantic model the pydantic-ai version returned, validated here, so
callers and artifacts are unchanged.

It does change one thing that has to be paid for by hand: **observability**. The
pipeline agents are autologged into MLflow because pydantic-ai runs in this
process; the SDK runs the `claude` CLI as a subprocess, so no autologger can see
a teacher call and none of it reaches the Traces tab on its own. `run_structured`
is the one place every Agent SDK call passes through, so the span is opened here
— reply, model, turns, tools, tokens and cost — and a second call path to the SDK
would need the same, or it would record nothing at all.

The prompts themselves are stored **by reference** (`prompt_refs`): a system
prompt is a constant repeated on every span of a run, and the writer's prompt
carries a whole subagent prompt plus its whole failure history. One copy lives on
the run, the span carries an id and a hash, and `resolve` reconstructs the exact
text or refuses.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# A short head of the user prompt goes on the span so a trace is readable at a
# glance without resolving anything. The *whole* prompt does not: see
# `prompt_refs` for what is stored instead and how to get the text back.
TRACED_HEAD_CHARS = 400


class TeacherCallError(RuntimeError):
    """The SDK returned nothing that validates against the requested schema."""


def _extract_json(text: str) -> Any:
    """Pull one JSON object out of a reply that may be wrapped in prose or fences."""
    fenced = text.split("```")
    for chunk in [text, *fenced] if len(fenced) > 1 else [text]:
        candidate = chunk.strip().removeprefix("json").strip()
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
    raise TeacherCallError(f"no JSON object in the reply: {text[:400]!r}")


async def run_structured(
    prompt: str,
    *,
    schema: type[T],
    system_prompt: str,
    mcp_servers: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    max_turns: int = 12,
    attempts: int = 2,
    refs: dict[str, Any] | None,
) -> tuple[T, dict[str, Any]]:
    """Run one Agent SDK turn and validate its reply against `schema`.

    `refs` says how to reconstruct this call's prompts — see `prompt_refs`. The
    span records those references and a short head rather than tens of kilobytes
    of text identical on every other span of the run.

    It is **required and has no default**, deliberately. Since the span no longer
    stores the prompt text, a call site that forgot to pass refs would record a
    prompt that is neither included nor recoverable — strictly worse than the
    text dump this replaced. Requiring the argument makes that a type error at
    the new call site instead of a discovery weeks later in the Traces tab. Pass
    an explicit `None` if a call genuinely has nothing worth referencing.

    Retries once by default, because the observed failure mode is transient: a
    call returns no content at all, and the next identical call succeeds. One
    such call out of fifty was enough to abort a whole cycle and discard twenty
    minutes of diagnosis, which is too fragile for something that runs
    unattended.

    Returns the parsed object and a usage dict — token counts and the tools it
    actually called. Usage is returned rather than logged here so the caller can
    put it on the right MLflow run; the campaign's throughput limit is teacher
    calls, so an unrecorded one is a gap in the only number that constrains it.
    """
    from convfinqa.evalloop import prompt_refs
    from convfinqa.llm import LM_TEACHER_MODEL
    from convfinqa.tracking import tracing

    last: Exception | None = None
    n = max(1, attempts)
    for attempt in range(n):
        # One span per attempt, so a transient empty reply is visible as a
        # failed call followed by a successful one rather than disappearing
        # into a retry loop that reports only its final result.
        with tracing.span(
            f"agent_sdk {schema.__name__}",
            span_type="LLM",
            attributes={
                "model": LM_TEACHER_MODEL,
                "schema": schema.__name__,
                "attempt": attempt + 1,
                "max_attempts": n,
                "max_turns": max_turns,
                "allowed_tools": list(allowed_tools or []),
            },
        ) as span:
            span.inputs(
                {
                    # References, not text. A prompt truncated to fit a trace is
                    # neither cheap nor faithful; a reference plus a hash is
                    # both, and `prompt_refs.resolve` turns it back into the
                    # exact bytes — or says why it cannot.
                    "refs": refs,
                    "system_prompt_sha": prompt_refs.sha(system_prompt),
                    "prompt_sha": prompt_refs.sha(prompt),
                    "prompt_chars": len(prompt),
                    "prompt_head": prompt[:TRACED_HEAD_CHARS],
                }
            )
            try:
                parsed, usage = await _run_structured_once(
                    prompt,
                    schema=schema,
                    system_prompt=system_prompt,
                    mcp_servers=mcp_servers,
                    allowed_tools=allowed_tools,
                    max_turns=max_turns,
                )
            except TeacherCallError as exc:
                span.set(error=repr(exc))
                last = exc
                if attempt + 1 < n:
                    await asyncio.sleep(2.0 * (attempt + 1))
                continue
            span.outputs(parsed.model_dump())
            tokens = usage.get("usage") or {}
            span.set(
                duration_ms=usage.get("duration_ms"),
                num_turns=usage.get("num_turns"),
                tools_used=usage.get("tools_used"),
                total_cost_usd=usage.get("total_cost_usd"),
                input_tokens=tokens.get("input_tokens"),
                output_tokens=tokens.get("output_tokens"),
                cache_read_input_tokens=tokens.get("cache_read_input_tokens"),
            )
            return parsed, usage
    raise last if last else TeacherCallError("no attempt was made")


async def _run_structured_once(
    prompt: str,
    *,
    schema: type[T],
    system_prompt: str,
    mcp_servers: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    max_turns: int = 12,
) -> tuple[T, dict[str, Any]]:
    """One attempt. See `run_structured` for the contract."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )

    from convfinqa.llm import teacher_options

    options = teacher_options(
        system_prompt=system_prompt,
        output_schema=schema.model_json_schema(),
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )
    if mcp_servers:
        options.mcp_servers = mcp_servers

    texts: list[str] = []
    tools_used: list[str] = []
    usage: dict[str, Any] = {}
    structured: Any = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tools_used.append(block.name)
            elif isinstance(message, ResultMessage):
                usage = {
                    "duration_ms": getattr(message, "duration_ms", None),
                    "num_turns": getattr(message, "num_turns", None),
                    "usage": getattr(message, "usage", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                }
                structured = getattr(message, "structured_result", None) or getattr(
                    message, "result", None
                )

    payload: Any = structured
    if isinstance(payload, str):
        payload = _extract_json(payload)
    if not isinstance(payload, dict):
        if not texts:
            raise TeacherCallError("the SDK returned no content at all")
        payload = _extract_json("\n".join(texts))

    try:
        parsed = schema.model_validate(payload)
    except ValidationError as exc:  # noqa: TRY302 — re-raised with the payload attached
        raise TeacherCallError(
            f"reply did not match {schema.__name__}: {exc}\npayload={payload!r}"
        ) from exc

    usage["tools_used"] = tools_used
    return parsed, usage
