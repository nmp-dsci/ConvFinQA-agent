"""The confidence judge in the serving path (s12).

A served agent_sdk turn is passed through the `judge_champion` prompt before
its answer is released. The judge reads the turn's capture — the same capture
the eval CSV row is built from — so `evalloop.judge.judge_payload` builds the
exact input it was scored on.

Fails **closed**: a judge that cannot run returns a `low` band with the error
in the record. A guard-rail that fails open is not a guard-rail, and the
record says why the turn was withheld.
"""

from __future__ import annotations

from typing import Any

from convfinqa.tracking import tracing


def judge_version() -> str | None:
    """The judge to serve, or None when the feature is off or none is registered."""
    from convfinqa.config import settings
    from convfinqa.tracking import registry

    if not settings.judge_enabled:
        return None
    return registry.judge_champion()


async def judge_capture(
    capture: dict[str, Any],
    *,
    report_id: str,
    turn_index: int,
    question: str,
    answer: str,
    program: str,
    version: str,
) -> dict[str, Any]:
    """Judge one served turn from its capture; return the verdict record."""
    from convfinqa.evalloop import judge

    row = judge.row_from_capture(
        capture,
        report_id=report_id,
        turn_index=turn_index,
        question=question,
        answer=answer,
        program=program,
    )
    with tracing.span(
        f"judge q{turn_index}",
        span_type="AGENT",
        attributes={
            "report_id": report_id,
            "turn_index": turn_index,
            "judge_version": version,
        },
    ) as span:
        try:
            verdict, usage = await judge.judge_turn(row, version=version)
        except Exception as exc:  # noqa: BLE001 — fail closed, and say so
            span.set(error=repr(exc), band="low")
            return {
                "version": version,
                "band": "low",
                "p_correct": 0.0,
                "reason": "judge unavailable — answer withheld",
                "checks": {},
                "metrics": {},
                "error": repr(exc),
            }
        record = judge.verdict_record(verdict, usage)
        span.set(band=record["band"], p_correct=record["p_correct"])
        span.outputs(record)
    return {"version": version, **record, "error": ""}
