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
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


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
) -> tuple[T, dict[str, Any]]:
    """Run one Agent SDK turn and validate its reply against `schema`.

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
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await _run_structured_once(
                prompt,
                schema=schema,
                system_prompt=system_prompt,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tools,
                max_turns=max_turns,
            )
        except TeacherCallError as exc:
            last = exc
            if attempt + 1 < max(1, attempts):
                await asyncio.sleep(2.0 * (attempt + 1))
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
