"""One recorded conversation → one training row (s28, row shape A).

A row is tau-telecom-sft's shape: ``{"messages", "tools", "meta"}`` in the
OpenAI chat format, rendered by the student's chat template at training time
with the loss on assistant messages only. Two things differ from tau:

- **The reasoning is kept.** DeepSeek's thinking goes into the assistant
  message as a visible ``<reasoning>`` block ahead of the call it leads to
  (`prompt.reasoning_block`). The student has no hidden thinking, so the
  template never strips it: earlier turns keep their reasoning in training and
  at serving alike.
- **N1, the detour, is collapsed.** DeepSeek sometimes writes its final answer
  as text and stops; Pydantic AI refuses that ("Please include your response
  in a tool call.") and DeepSeek re-sends it as a ``StructuredOutput`` call,
  often after a line of reasoning about the refusal. The row keeps the call,
  keeps the reasoning that produced the answer (the refused reply's), and
  drops the refused text, the retry prompt and the reasoning about it — after
  checking the re-sent answer is the refused one.

The system message and tool JSON are not reconstructed: `wire` sends one
request through the student's own model and harness to a capturing transport
and reads them off it, so a row carries the bytes the student is served.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
from typing import Any

from convfinqa.sft import trajectories as tj
from convfinqa.sft.prompt import reasoning_block

OUTPUT_TOOL = "StructuredOutput"
REFUSAL = "Please include your response in a tool call."


class RowRejectedError(Exception):
    """A conversation the builder will not train on, and the rule it failed."""

    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


@functools.cache
def wire(version: str = "sdk_v1") -> dict[str, Any]:
    """The system message and tools exactly as the student is sent them."""
    import httpx

    from convfinqa.backends import open_agent
    from convfinqa.llm import student_wire_model
    from convfinqa.sft.prompt import student_prompt

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("body", json.loads(request.content))
        answer = {
            "turn_type": "number",
            "conv_type": "Type I",
            "answer": "0",
            "reasoning": "-",
        }
        call = {"name": OUTPUT_TOOL, "arguments": json.dumps(answer)}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_0", "type": "function", "function": call}],
        }
        return httpx.Response(
            200,
            json={
                "id": "wire",
                "object": "chat.completion",
                "created": 0,
                "model": "student",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls"}
                ],
            },
        )

    model = student_wire_model(httpx.MockTransport(handler))
    agent = open_agent.build_agent(student_prompt(version), model, [])
    asyncio.run(agent.run("wire"))
    body = seen["body"]
    system = [m for m in body["messages"] if m.get("role") == "system"]
    if len(system) != 1:
        raise RuntimeError(
            f"expected one system message on the wire, saw {len(system)}"
        )
    return {"system": str(system[0]["content"]), "tools": list(body["tools"])}


def _as_text(content: Any) -> str:
    return (
        content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _json_of(text: str) -> dict[str, Any] | None:
    """The JSON object a refused text answer holds, fences and all."""
    try:
        loaded = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError:
        return None
    return dict(loaded) if isinstance(loaded, dict) else None


def _decision(answer: dict[str, Any]) -> tuple[Any, ...]:
    """What a final answer decides: its type, plan, operands' values and answer.

    DeepSeek re-generates the answer it re-sends, so the free-text fields
    (``reasoning``, each operand's ``source`` and question wording) are
    reworded; the decision must not change. The call's own arguments, which
    are what the harness scored, are what the row keeps.
    """
    retrieved = answer.get("retrieved") or []
    values = tuple(str(r.get("answer")) for r in retrieved if isinstance(r, dict))
    return (
        answer.get("turn_type"),
        answer.get("conv_type"),
        str(answer.get("program") or ""),
        values,
        str(answer.get("answer")),
    )


def turn_messages(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """A turn's assistant and tool messages, N1 applied; and how many detours."""
    out: list[dict[str, Any]] = []
    held: dict[str, Any] | None = None  # a text-only reply, waiting for its refusal
    refused: dict[str, Any] | None = None  # the refused reply, waiting for its call
    detours = 0
    for message in steps:
        if message.get("kind") == "response":
            thinking = "\n\n".join(
                str(p.get("content") or "").strip()
                for p in tj.parts(message, "thinking")
                if str(p.get("content") or "").strip()
            )
            text = "".join(
                str(p.get("content") or "") for p in tj.parts(message, "text")
            )
            calls = tj.parts(message, "tool-call")
            if held is not None:
                raise RowRejectedError(
                    "N1", "a text reply was followed by another reply, not a refusal"
                )
            if not calls:
                held = {"thinking": thinking, "text": text.strip()}
                continue
            if refused is not None:
                # the reply that re-sends the refused answer: its own reasoning is
                # about the refusal, so the refused reply's reasoning stands
                if [c.get("tool_name") for c in calls] != [OUTPUT_TOOL]:
                    raise RowRejectedError(
                        "N1",
                        "the reply after a refusal is not one StructuredOutput call",
                    )
                if _decision(refused["answer"]) != _decision(tj.tool_args(calls[0])):
                    raise RowRejectedError(
                        "N1",
                        "the re-sent answer decides differently from the refused text",
                    )
                thinking, text = refused["thinking"], ""
                refused = None
            content = reasoning_block(thinking) if thinking else ""
            if text.strip():
                content = f"{content}\n{text.strip()}" if content else text.strip()
            out.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": str(c.get("tool_call_id") or f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": str(c["tool_name"]),
                                "arguments": json.dumps(
                                    tj.tool_args(c), ensure_ascii=False
                                ),
                            },
                        }
                        for i, c in enumerate(calls)
                    ],
                }
            )
            continue
        for part in message.get("parts") or []:
            kind = part.get("part_kind")
            if kind == "tool-return":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(part.get("tool_call_id") or ""),
                        "content": _as_text(part.get("content")),
                    }
                )
            elif kind == "retry-feedback":
                if part.get("tool_name"):
                    raise RowRejectedError(
                        "F5", f"a {part['tool_name']} call was refused"
                    )
                if held is None or str(part.get("content") or "") != REFUSAL:
                    raise RowRejectedError(
                        "N1", "a retry prompt that is not the text-answer refusal"
                    )
                answer = _json_of(held["text"])
                if answer is None:
                    raise RowRejectedError(
                        "N1", "the refused text is not a JSON answer"
                    )
                refused = {"thinking": held["thinking"], "answer": answer}
                held = None
                detours += 1
            else:
                raise RowRejectedError("F3", f"an unexpected {kind} part inside a turn")
    if held is not None or refused is not None:
        raise RowRejectedError("N1", "a text answer was never re-sent as a call")
    return out, detours


def to_row(
    report_id: str, messages: list[dict[str, Any]], *, version: str = "sdk_v1"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One whole conversation as a row; raises `RowRejectedError` when it cannot be one."""
    w = wire(version)
    rendered: list[dict[str, Any]] = [{"role": "system", "content": w["system"]}]
    turns = tj.turns(messages)
    detours = 0
    for index, turn in enumerate(turns):
        rendered.append({"role": "user", "content": turn["prompt"]})
        body, n = turn_messages(turn["steps"])
        detours += n
        replies = [m for m in body if m["role"] == "assistant"]
        names = (
            [c["function"]["name"] for c in replies[-1]["tool_calls"]]
            if replies
            else []
        )
        if names != [OUTPUT_TOOL]:
            raise RowRejectedError(
                "F3", f"q{index} does not end on one StructuredOutput call"
            )
        rendered += body
    meta = {"report_id": report_id, "n_turns": len(turns), "n_detours": detours}
    return {"messages": rendered, "tools": w["tools"], "meta": meta}, meta
