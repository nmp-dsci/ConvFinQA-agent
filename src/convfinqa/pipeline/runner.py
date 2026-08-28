"""Run / stream a single conversation turn end-to-end through the 4-stage pipeline.

There is exactly one implementation of the pipeline flow: `turn_events`, an async
generator that yields a typed event per stage. `stream_turn` forwards those events
to SSE; `run_turn` drains them and returns the final answer. Previously the two
paths were separate transcriptions of the same four stages, which is how they
drifted — the streaming path never populated `capture`, so a turn watched live
produced no trace while the same turn scored in an eval produced a full one.

The event vocabulary is the frontend's contract and the demo pack's format:

    stage_start   {stage}
    stage_output  {stage, output, metrics}
    tool_call     {stage, tool, args}
    tool_return   {stage, tool, result}
    answer        {answer, program}
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from convfinqa.backends.pydantic import default_agents
from convfinqa.data.loader import _DOCS
from convfinqa.data.schemas import ConversationHistory
from convfinqa.llm import call_with_budget
from convfinqa.pipeline.wire_format import render_chat_inputs as _render_chat_inputs


def _coerce_args(args: Any) -> Any:
    """Render a ToolCallPart's args as plain dict/string for JSON transport."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def _tool_events(messages: list[Any], stage: str) -> list[dict[str, Any]]:
    """Replay tool-call / tool-return parts from a finished agent run."""
    events: list[dict[str, Any]] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                events.append(
                    {
                        "event": "tool_call",
                        "stage": stage,
                        "tool": part.tool_name,
                        "args": _coerce_args(part.args),
                    }
                )
            elif isinstance(part, ToolReturnPart):
                events.append(
                    {
                        "event": "tool_return",
                        "stage": stage,
                        "tool": part.tool_name,
                        "result": str(part.content),
                    }
                )
    return events


def _calc_trajectory(messages: list[Any]) -> list[dict[str, Any]]:
    """Render a calculator's tool-call trace as a JSON-friendly list of events."""
    return [
        {k: v for k, v in event.items() if k != "stage"}
        for event in _tool_events(messages, "calculator")
    ]


def _usage_of(result: Any) -> dict[str, int]:
    """Extract token counts from a run result, tolerating provider shape drift."""
    try:
        usage = result.usage()
    except Exception:  # noqa: BLE001 — usage is telemetry; never fail a turn for it
        return {}
    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


async def _run_stage(
    agent: Agent[None, Any], message: Any
) -> tuple[Any, dict[str, Any]]:
    """Run one agent under the global call budget; return (result, stage metrics)."""
    started = time.perf_counter()
    result = await call_with_budget(lambda: agent.run(message))
    metrics: dict[str, Any] = {
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        **_usage_of(result),
    }
    return result, metrics


async def turn_events(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
    *,
    agents: dict[str, Agent[None, Any]] | None = None,
    capture: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn, yielding an event per stage and appending to `conversation`.

    `capture`, when supplied, is filled with the same per-stage IO the eval CSVs
    record — so a streamed turn and a scored turn produce identical traces.
    """
    document = _DOCS[report_id]
    hist_text = conversation.as_text()
    resolved = agents or default_agents()
    tri = resolved["triage"]
    pre = resolved["preprocess"]
    ret = resolved["retriever"]
    calc_a = resolved["calculator"]

    if capture is not None:
        capture["history_text"] = hist_text

    with logfire.span(
        "turn {report_id}",
        report_id=report_id,
        question=question,
        history_turns=len(conversation.pairs),
    ) as span:
        # ---- Stage 1: triage -------------------------------------------------
        yield {"event": "stage_start", "stage": "triage"}
        triage_input: dict[str, Any] = {"question": question}
        triage_result, triage_metrics = await _run_stage(
            tri, _render_chat_inputs(triage_input)
        )
        triage = triage_result.output
        span.set_attribute("turn_type", triage.turn_type)
        span.set_attribute("conv_type", triage.conv_type)
        if capture is not None:
            capture["triage"] = {
                "input": triage_input,
                "output": triage.model_dump(),
                "reasoning": getattr(triage, "reasoning", "") or "",
                "metrics": triage_metrics,
            }
        yield {
            "event": "stage_output",
            "stage": "triage",
            "output": triage.model_dump(),
            "metrics": triage_metrics,
        }

        # ---- Number path: retrieve the value and stop -------------------------
        if triage.turn_type == "number":
            yield {"event": "stage_start", "stage": "retriever"}
            retr_input: dict[str, Any] = {
                "turn_type": "number",
                "questions": [question],
                "history": hist_text,
            }
            retr_result, retr_metrics = await _run_stage(
                ret, _render_chat_inputs({**retr_input, "document": document})
            )
            retrieved = retr_result.output
            answer = str(retrieved.answers[0].answer)
            if capture is not None:
                capture["retriever"] = {
                    "input": retr_input,
                    "output": retrieved.model_dump(),
                    "reasoning": getattr(retrieved, "reasoning", "") or "",
                    "metrics": retr_metrics,
                }
                capture["preprocess"] = None
                capture["calculator"] = None
            yield {
                "event": "stage_output",
                "stage": "retriever",
                "output": retrieved.model_dump(),
                "metrics": retr_metrics,
            }
            span.set_attribute("answer", answer)
            conversation.append(question=question, answer=answer, report_id=report_id)
            yield {"event": "answer", "answer": answer, "program": ""}
            return

        # ---- Program path: preprocess → retrieve → calculate -------------------
        yield {"event": "stage_start", "stage": "preprocess"}
        pp_input: dict[str, Any] = {
            "question": question,
            "history": hist_text,
            "conv_type": triage.conv_type,
        }
        pp_result, pp_metrics = await _run_stage(pre, _render_chat_inputs(pp_input))
        preprocess = pp_result.output
        span.set_attribute("program", preprocess.program)
        if capture is not None:
            capture["preprocess"] = {
                "input": pp_input,
                "output": preprocess.model_dump(),
                "reasoning": getattr(preprocess, "reasoning", "") or "",
                "metrics": pp_metrics,
            }
        yield {
            "event": "stage_output",
            "stage": "preprocess",
            "output": preprocess.model_dump(),
            "metrics": pp_metrics,
        }

        yield {"event": "stage_start", "stage": "retriever"}
        prog_retr_input: dict[str, Any] = {
            "turn_type": "program",
            "questions": list(preprocess.sub_questions),
            "history": hist_text,
        }
        retr_result, retr_metrics = await _run_stage(
            ret, _render_chat_inputs({**prog_retr_input, "document": document})
        )
        retrieved = retr_result.output
        if capture is not None:
            capture["retriever"] = {
                "input": prog_retr_input,
                "output": retrieved.model_dump(),
                "reasoning": getattr(retrieved, "reasoning", "") or "",
                "metrics": retr_metrics,
            }
        yield {
            "event": "stage_output",
            "stage": "retriever",
            "output": retrieved.model_dump(),
            "metrics": retr_metrics,
        }

        yield {"event": "stage_start", "stage": "calculator"}
        calc_input: dict[str, Any] = {
            "question": question,
            "retrieved": [qa.model_dump() for qa in retrieved.answers],
            "program": preprocess.program,
        }
        calc_result, calc_metrics = await _run_stage(
            calc_a, _render_chat_inputs(calc_input)
        )
        for event in _tool_events(calc_result.all_messages(), "calculator"):
            yield event
        calc = calc_result.output
        answer = str(calc.answer)
        if capture is not None:
            capture["calculator"] = {
                "input": calc_input,
                "output": calc.model_dump(),
                "trajectory": _calc_trajectory(calc_result.all_messages()),
                "metrics": calc_metrics,
            }
        yield {
            "event": "stage_output",
            "stage": "calculator",
            "output": calc.model_dump(),
            "metrics": calc_metrics,
        }
        span.set_attribute("answer", answer)
        conversation.append(question=question, answer=answer, report_id=report_id)
        yield {"event": "answer", "answer": answer, "program": preprocess.program}


async def run_turn(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
    *,
    agents: dict[str, Agent[None, Any]] | None = None,
    capture: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Run one turn end-to-end, append answer to conversation; return (answer, program).

    Drains the same generator the streaming path uses — the batch and live paths
    cannot diverge, because there is only one of them.
    """
    answer = ""
    program = ""
    async for event in turn_events(
        question, report_id, conversation, agents=agents, capture=capture
    ):
        if event.get("event") == "answer":
            answer = str(event.get("answer", ""))
            program = str(event.get("program", ""))
    return answer, program


async def stream_turn(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
    *,
    agents: dict[str, Agent[None, Any]] | None = None,
    capture: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn and yield event dicts as each stage completes."""
    async for event in turn_events(
        question, report_id, conversation, agents=agents, capture=capture
    ):
        yield event


class ConversationRunner:
    """Walk all turns of one conversation, threading history."""

    async def run_conversation(
        self,
        report_id: str,
        questions: list[str],
        *,
        agents: dict[str, Agent[None, Any]] | None = None,
        captures: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Walk all turns of one conversation in order, threading history."""
        conversation = ConversationHistory()
        preds: list[str] = []
        programs: list[str] = []
        for question in questions:
            cap: dict[str, Any] = {}
            answer, program = await run_turn(
                question, report_id, conversation, agents=agents, capture=cap
            )
            preds.append(answer)
            programs.append(program)
            if captures is not None:
                captures.append(cap)
        return preds, programs
