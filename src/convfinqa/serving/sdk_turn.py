"""The agent_sdk runtime's turn as the chat's event stream (s12).

`pipeline.runner.turn_events` yields a frame per stage as each stage finishes.
The single-session runtime finishes all four at once — the session returns one
structured reply — so the same frames are emitted from the capture after the
reply lands, in the same order and with the same payload keys, followed by two
frames the pipeline never emits:

- ``judge`` — the confidence verdict (band, p_correct, checks, reason), when a
  judge is enabled and registered;
- ``answer`` carries ``band`` and ``withheld``. What a ``low`` band does is
  `settings.judge_mode`: under ``advisory`` (the default) the answer is
  released with the band beside it, and the UI shows the judge's reason and
  its failed checks as a caveat; under ``gate`` the answer is withheld —
  ``answer`` is empty on the wire and in the visible history, the stage
  frames that carry it are redacted, and the turn's record
  (`capture["judge"]`, the trace row) keeps the value the judge refused to
  release, so withheld answers stay measurable.

The default is advisory because gating was measured and did not earn its
cost: it withholds four correct answers for every wrong one it stops, for a
released-set accuracy indistinguishable from releasing everything (s13).

The session itself keeps the real answer in its own context, so later turns
that depend on a withheld one are still coherent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from convfinqa.serving.sessions import SessionState

WITHHELD_TEXT = ""


def _redact_content_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Strip the reasoning trail's content, keeping only stage/event/metrics.

    A `stage_output` frame carries the stage's `output` (which, for the
    retriever on a number turn or the calculator on a program turn, *is* the
    final answer); `tool_call`/`tool_return` frames carry the arithmetic that
    produced it. On a withheld turn none of that may reach the wire, exactly
    as the answer frame itself is emptied.
    """
    redacted = dict(frame)
    if "output" in redacted:
        redacted["output"] = {}
    if "args" in redacted:
        redacted["args"] = {}
    if "result" in redacted:
        redacted["result"] = ""
    return redacted


def stage_frames(capture: dict[str, Any]) -> list[dict[str, Any]]:
    """The four stage frames a capture stands for, in pipeline order."""
    frames: list[dict[str, Any]] = []
    triage = capture.get("triage") or {}
    frames.append({"event": "stage_start", "stage": "triage"})
    if triage.get("output"):
        frames.append(
            {
                "event": "stage_output",
                "stage": "triage",
                "output": triage["output"],
                "metrics": triage.get("metrics", {}) or {},
            }
        )
    preprocess = capture.get("preprocess") or {}
    if preprocess.get("output"):
        frames.append({"event": "stage_start", "stage": "preprocess"})
        frames.append(
            {
                "event": "stage_output",
                "stage": "preprocess",
                "output": preprocess["output"],
                "metrics": preprocess.get("metrics", {}) or {},
            }
        )
    retriever = capture.get("retriever") or {}
    frames.append({"event": "stage_start", "stage": "retriever"})
    if retriever.get("output"):
        frames.append(
            {
                "event": "stage_output",
                "stage": "retriever",
                "output": retriever["output"],
                "metrics": retriever.get("metrics", {}) or {},
            }
        )
    calculator = capture.get("calculator") or {}
    if calculator.get("output"):
        frames.append({"event": "stage_start", "stage": "calculator"})
        for step in calculator.get("trajectory", []) or []:
            if isinstance(step, dict) and step.get("event") in {
                "tool_call",
                "tool_return",
            }:
                frames.append({**step, "stage": "calculator"})
        frames.append(
            {
                "event": "stage_output",
                "stage": "calculator",
                "output": calculator["output"],
                "metrics": calculator.get("metrics", {}) or {},
            }
        )
    return frames


async def sdk_turn_events(
    question: str,
    state: SessionState,
    *,
    capture: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Run one turn on the session's live SDK client; yield the event stream."""
    from convfinqa.serving import judge as serving_judge
    from convfinqa.serving.sdk_session import SdkSession

    if state.sdk_session is None:
        state.sdk_session = SdkSession(state.report_id)
    session: SdkSession = state.sdk_session
    hist_text = state.conversation.as_text()
    turn_index = len(state.conversation.pairs)

    yield {"event": "stage_start", "stage": "triage"}
    result, cap, _usage = await session.ask(question, history_text=hist_text)
    capture.update(cap)
    answer = result.answer
    program = result.program if result.turn_type == "program" else ""
    # Held, not streamed yet: the single-session runtime answers all four
    # stages in one reply, so there is no progressive output to preserve, and
    # these frames carry the answer the judge has not yet ruled on.
    content_frames = stage_frames(capture)[1:]

    band: str | None = None
    verdict: dict[str, Any] | None = None
    version = serving_judge.judge_version()
    if version:
        verdict = await serving_judge.judge_capture(
            capture,
            report_id=state.report_id,
            turn_index=turn_index,
            question=question,
            answer=answer,
            program=program,
            version=version,
        )
        capture["judge"] = {**verdict, "answer": answer}
        band = str(verdict["band"])

    # `advisory` (the default) shows the answer with the judge's caution beside
    # it; only `gate` withholds. See `settings.judge_mode` for why the
    # measurement puts the default here.
    from convfinqa.config import settings

    withheld = band == "low" and settings.judge_mode == "gate"
    for frame in content_frames:
        yield _redact_content_frame(frame) if withheld else frame

    if verdict is not None:
        yield {"event": "stage_start", "stage": "judge"}
        yield {"event": "judge", **verdict}

    shown = WITHHELD_TEXT if withheld else answer
    state.conversation.append(
        question=question, answer=shown, report_id=state.report_id
    )
    yield {
        "event": "answer",
        "answer": shown,
        "program": program,
        "band": band,
        "withheld": withheld,
    }
