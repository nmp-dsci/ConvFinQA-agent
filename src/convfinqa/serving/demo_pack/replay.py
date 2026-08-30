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
from convfinqa.serving.demo_pack.store import DemoPack, PackedTurn, load_pack, normalise

# Per-event pacing, in seconds. Stage boundaries carry the sense of work being
# done; tool calls inside the calculator loop are quicker because they were.
_DELAYS = {
    "stage_start": 0.45,
    "stage_output": 0.35,
    "tool_call": 0.18,
    "tool_return": 0.12,
    "answer": 0.2,
}

# Replay's own match floor, well below the pack's strict `MATCH_THRESHOLD`.
#
# The strict threshold was tuned for a silent substitution — if the visitor is
# never told which recording answered them, the only safe match is one they
# would have written themselves. But a visitor who types their own phrasing and
# gets a 404 concludes the demo is broken, which is worse than the risk this was
# guarding against. Answering a loose match *and naming what was matched* is the
# honest trade: the visitor can see the swap and judge it, and the banner is what
# buys the extra headroom. Below this floor the pack still declines outright —
# a wrong number presented as an answer remains the failure that must not happen.
FUZZY_THRESHOLD = 0.35

# Every question in this corpus opens "what was the ..." / "what is the ...", so
# a low threshold on its own rewards the boilerplate: "what is the weather today"
# scores 0.44 against "what is the net change from 2009 to 2010?" purely on words
# that carry no meaning. The threshold measures overall similarity; this measures
# whether any of it was *about anything*, and a match needs both.
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "many",
        "much",
        "of",
        "on",
        "or",
        "that",
        "the",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)


def _content_words(question: str) -> set[str]:
    """Tokens that carry subject matter rather than question boilerplate."""
    return {word for word in normalise(question).split() if word not in _STOPWORDS}


def shares_subject(asked: str, recorded: str) -> bool:
    """True when the two questions have at least one content word in common.

    A recorded question with no content words at all (there are none, but the
    pack is data) would otherwise be unmatchable, so it falls back to allowing
    the similarity score to decide on its own.
    """
    recorded_words = _content_words(recorded)
    if not recorded_words:
        return True
    return bool(_content_words(asked) & recorded_words)


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


def resolve(
    report_id: str, question: str, pack: DemoPack | None = None
) -> tuple[PackedTurn, float, bool]:
    """Resolve `question` to a recorded turn, or raise `NoRecordingError`.

    Returns the turn, its score, and whether the question was matched *exactly* —
    the last of which decides whether the caller owes the visitor a banner.
    """
    resolved = pack if pack is not None else load_pack()
    turn, score = resolved.nearest(report_id, question)
    if turn is None or score < FUZZY_THRESHOLD:
        raise NoRecordingError(report_id, question, score)
    exact = normalise(turn.question) == normalise(question)
    if not exact and not shares_subject(question, turn.question):
        raise NoRecordingError(report_id, question, score)
    return turn, score, exact


def capture_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the `capture` dict a live turn produces from a recorded stream.

    Without this a demo turn reaches the trace store with an empty capture and
    lands in the metrics as a turn that cost nothing and took no time — zeros
    that read as facts. The recorded `metrics` payloads are the real numbers from
    when the turn actually ran, so replaying them is the honest thing to store;
    what must never be stored is the *replay's* four-second pacing.
    """
    capture: dict[str, Any] = {}
    trajectory: list[dict[str, Any]] = []
    for event in events:
        name = str(event.get("event"))
        stage = str(event.get("stage", ""))
        if name == "stage_output" and stage:
            capture[stage] = {
                "output": event.get("output", {}),
                "metrics": event.get("metrics", {}) or {},
            }
        elif name in {"tool_call", "tool_return"} and stage == "calculator":
            trajectory.append({k: v for k, v in event.items() if k != "stage"})
    if trajectory and isinstance(capture.get("calculator"), dict):
        capture["calculator"]["trajectory"] = trajectory
    return capture


async def replay_turn(
    question: str,
    report_id: str,
    conversation: ConversationHistory,
    *,
    pack: DemoPack | None = None,
    pace: bool = True,
    capture: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the recorded event stream for `question`, pacing it like a live turn.

    Appends to `conversation` exactly as the live path does, so multi-turn
    history in the demo behaves the way it does in dev. When the question was
    only *approximately* matched, a `matched` event leads the stream so the UI
    can say which recording it is showing rather than implying it answered the
    words that were typed.
    """
    turn, score, exact = resolve(report_id, question, pack)

    if not exact:
        yield {
            "event": "matched",
            "matched_question": turn.question,
            "asked_question": question,
            "score": round(score, 4),
        }

    for event in turn.events:
        if pace:
            await asyncio.sleep(_DELAYS.get(str(event.get("event")), 0.2))
        yield event

    if capture is not None:
        capture["history_text"] = conversation.as_text()
        capture.update(capture_from_events(turn.events))
        capture["replayed"] = True
        capture["matched_question"] = turn.question

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
