"""Run / stream a single conversation turn end-to-end through the 4-stage pipeline."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from convfinqa.backends.pydantic import (
    calculator_agent,
    preprocess_agent,
    retriever_agent,
    triage_agent,
)
from convfinqa.data.loader import _DOCS
from convfinqa.data.schemas import ConversationHistory
from convfinqa.pipeline.wire_format import render_chat_inputs as _render_chat_inputs


def _calc_trajectory(messages: list[Any]) -> list[dict[str, Any]]:
    """Render a calculator's tool-call trace as a JSON-friendly list of events."""
    events: list[dict[str, Any]] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart):
                args = part.args
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                events.append(
                    {"event": "tool_call", "tool": part.tool_name, "args": args}
                )
            elif isinstance(part, ToolReturnPart):
                events.append(
                    {
                        "event": "tool_return",
                        "tool": part.tool_name,
                        "result": str(part.content),
                    }
                )
    return events


async def run_turn(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
    *,
    agents: dict[str, Agent] | None = None,
    capture: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Run one turn end-to-end, append answer to conversation; return (answer, program)."""
    document = _DOCS[report_id]
    hist_text = conversation.as_text()

    tri = (agents or {}).get("triage", triage_agent)
    pre = (agents or {}).get("preprocess", preprocess_agent)
    ret = (agents or {}).get("retriever", retriever_agent)
    calc_a = (agents or {}).get("calculator", calculator_agent)

    if capture is not None:
        capture["history_text"] = hist_text

    triage_input = {"question": question}
    triage_msg = _render_chat_inputs(triage_input)
    triage = (await tri.run(triage_msg)).output

    if capture is not None:
        capture["triage"] = {
            "input": triage_input,
            "output": triage.model_dump(),
            "reasoning": getattr(triage, "reasoning", "") or "",
        }

    if triage.turn_type == "number":
        retr_input_csv = {
            "turn_type": "number",
            "questions": [question],
            "history": hist_text,
        }
        retr_msg = _render_chat_inputs({**retr_input_csv, "document": document})
        retrieved = (await ret.run(retr_msg)).output
        answer = str(retrieved.answers[0].answer)
        if capture is not None:
            capture["retriever"] = {
                "input": retr_input_csv,
                "output": retrieved.model_dump(),
                "reasoning": getattr(retrieved, "reasoning", "") or "",
            }
            capture["preprocess"] = None
            capture["calculator"] = None
        conversation.append(question=question, answer=answer, report_id=report_id)
        return answer, ""

    pp_input = {
        "question": question,
        "history": hist_text,
        "conv_type": triage.conv_type,
    }
    pp_msg = _render_chat_inputs(pp_input)
    preprocess = (await pre.run(pp_msg)).output
    if capture is not None:
        capture["preprocess"] = {
            "input": pp_input,
            "output": preprocess.model_dump(),
            "reasoning": getattr(preprocess, "reasoning", "") or "",
        }

    retr_input_csv = {
        "turn_type": "program",
        "questions": list(preprocess.sub_questions),
        "history": hist_text,
    }
    retr_msg = _render_chat_inputs({**retr_input_csv, "document": document})
    retrieved = (await ret.run(retr_msg)).output
    if capture is not None:
        capture["retriever"] = {
            "input": retr_input_csv,
            "output": retrieved.model_dump(),
            "reasoning": getattr(retrieved, "reasoning", "") or "",
        }

    calc_input = {
        "question": question,
        "retrieved": [qa.model_dump() for qa in retrieved.answers],
        "program": preprocess.program,
    }
    calc_msg = _render_chat_inputs(calc_input)
    calc_result = await calc_a.run(calc_msg)
    calc = calc_result.output
    answer = str(calc.answer)
    if capture is not None:
        capture["calculator"] = {
            "input": calc_input,
            "output": calc.model_dump(),
            "trajectory": _calc_trajectory(calc_result.all_messages()),
        }
    conversation.append(question=question, answer=answer, report_id=report_id)
    return answer, preprocess.program


def _coerce_args(args: Any) -> Any:
    """Render a ToolCallPart's args as plain dict/string for JSON transport."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def _tool_events_from_messages(messages: list[Any], stage: str) -> list[dict[str, Any]]:
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


async def stream_turn(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn and yield event dicts as each stage completes."""
    document = _DOCS[report_id]
    hist_text = conversation.as_text()

    with logfire.span(
        "stream_turn {report_id}",
        report_id=report_id,
        question=question,
        history_turns=len(conversation.pairs),
    ) as span:
        yield {"event": "stage_start", "stage": "triage"}
        triage_msg = _render_chat_inputs({"question": question})
        triage = (await triage_agent.run(triage_msg)).output
        span.set_attribute("turn_type", triage.turn_type)
        span.set_attribute("conv_type", triage.conv_type)
        yield {
            "event": "stage_output",
            "stage": "triage",
            "output": triage.model_dump(),
        }

        if triage.turn_type == "number":
            yield {"event": "stage_start", "stage": "retriever"}
            retr_msg = _render_chat_inputs(
                {
                    "turn_type": "number",
                    "questions": [question],
                    "document": document,
                    "history": hist_text,
                }
            )
            retrieved = (await retriever_agent.run(retr_msg)).output
            yield {
                "event": "stage_output",
                "stage": "retriever",
                "output": retrieved.model_dump(),
            }
            answer = str(retrieved.answers[0].answer)
            span.set_attribute("answer", answer)
            conversation.append(question=question, answer=answer, report_id=report_id)
            yield {"event": "answer", "answer": answer}
            return

        yield {"event": "stage_start", "stage": "preprocess"}
        pp_msg = _render_chat_inputs(
            {
                "question": question,
                "history": hist_text,
                "conv_type": triage.conv_type,
            }
        )
        preprocess = (await preprocess_agent.run(pp_msg)).output
        span.set_attribute("program", preprocess.program)
        yield {
            "event": "stage_output",
            "stage": "preprocess",
            "output": preprocess.model_dump(),
        }

        yield {"event": "stage_start", "stage": "retriever"}
        retr_msg = _render_chat_inputs(
            {
                "turn_type": "program",
                "questions": list(preprocess.sub_questions),
                "document": document,
                "history": hist_text,
            }
        )
        retrieved = (await retriever_agent.run(retr_msg)).output
        yield {
            "event": "stage_output",
            "stage": "retriever",
            "output": retrieved.model_dump(),
        }

        yield {"event": "stage_start", "stage": "calculator"}
        calc_msg = _render_chat_inputs(
            {
                "question": question,
                "retrieved": [qa.model_dump() for qa in retrieved.answers],
                "program": preprocess.program,
            }
        )
        calc_result = await calculator_agent.run(calc_msg)
        for ev in _tool_events_from_messages(calc_result.all_messages(), "calculator"):
            yield ev
        answer = str(calc_result.output.answer)
        span.set_attribute("answer", answer)
        conversation.append(question=question, answer=answer, report_id=report_id)
        yield {
            "event": "stage_output",
            "stage": "calculator",
            "output": {"answer": answer},
        }
        yield {"event": "answer", "answer": answer}


class ConversationRunner:
    """Walk all turns of one conversation, threading history."""

    async def run_conversation(
        self,
        report_id: str,
        questions: list[str],
        *,
        captures: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Walk all turns of one conversation in order, threading history."""
        conversation = ConversationHistory()
        preds: list[str] = []
        programs: list[str] = []
        for question in questions:
            cap: dict[str, Any] = {}
            answer, program = await run_turn(
                question, report_id, conversation, capture=cap
            )
            preds.append(answer)
            programs.append(program)
            if captures is not None:
                captures.append(cap)
        return preds, programs
