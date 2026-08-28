"""Replay a recorded turn as a live-looking SSE stream.

Pacing is the whole trick. A real turn takes 30–60 seconds because four model
calls happen in sequence; dumping the recorded events instantly would read as a
canned response, and sleeping the real duration would read as broken. A few
hundred milliseconds per stage gives the same *shape* — stages resolving one
after another, the calculator's tool loop ticking through — in about four
seconds.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from convfinqa.data.schemas import ConversationHistory
from convfinqa.serving.demo_pack.store import DemoPack, PackedTurn, load_pack

# Per-event pacing, in seconds. Stage boundaries carry the sense of work being
# done; tool calls inside the calculator loop are quicker because they were.
_DELAYS = {
    "stage_start": 0.45,
    "stage_output": 0.35,
    "tool_call": 0.18,
    "tool_return": 0.12,
    "answer": 0.2,
}


class NoRecordingError(LookupError):
    """Raised when the pack holds nothing close enough to the asked question."""

    code = "no_recording"

    def __init__(self, report_id: str, question: str, best_score: float) -> None:
        self.report_id = report_id
        self.question = question
        self.best_score = best_score
        super().__init__(
            "This demo replays recorded conversations and has no recording for "
            "that question. Pick one of the suggested questions for this report "
            "to see the agent work end to end."
        )


async def replay_turn(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
    *,
    pack: DemoPack | None = None,
    pace: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the recorded event stream for `question`, pacing it like a live turn.

    Appends to `conversation` exactly as the live path does, so multi-turn
    history in the demo behaves the way it does in dev.
    """
    resolved = pack if pack is not None else load_pack()
    turn, score = resolved.match(report_id, question)
    if turn is None:
        raise NoRecordingError(report_id, question, score)

    for event in turn.events:
        if pace:
            await asyncio.sleep(_DELAYS.get(str(event.get("event")), 0.2))
        yield event

    conversation.append(question=question, answer=turn.answer, report_id=report_id)


def suggested_questions(
    report_id: str, pack: DemoPack | None = None
) -> list[dict[str, Any]]:
    """The chip rail: what this report can actually answer in demo mode."""
    resolved = pack if pack is not None else load_pack()
    return [
        {
            "turn_index": turn.turn_index,
            "question": turn.question,
            "gold_answer": turn.gold_answer,
            "correct": turn.correct,
        }
        for turn in resolved.turns_for(report_id)
    ]


def packed_reports(pack: DemoPack | None = None) -> list[dict[str, Any]]:
    """Reports the demo can hold a conversation about, with their turn counts."""
    resolved = pack if pack is not None else load_pack()
    return [
        {
            "report_id": report_id,
            "n_questions": len(resolved.turns_for(report_id)),
        }
        for report_id in resolved.report_ids
    ]


def find_turn(
    report_id: str, question: str, pack: DemoPack | None = None
) -> PackedTurn | None:
    """Best recorded turn for a question, or None below the match threshold."""
    resolved = pack if pack is not None else load_pack()
    turn, _ = resolved.match(report_id, question)
    return turn
