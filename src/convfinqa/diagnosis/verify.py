"""Verify a candidate system_prompt patch by replaying turns 0..k."""

from __future__ import annotations

from typing import Any

from convfinqa.backends.pydantic import make_agents
from convfinqa.data.loader import qa_data
from convfinqa.data.schemas import ConversationHistory
from convfinqa.diagnosis.models import (
    AgentName,
    FailureReason,
    FixAttempt,
    StageIO,
    TurnResult,
)
from convfinqa.evaluation.metrics import numeric_match
from convfinqa.pipeline.runner import run_turn

PATCH_HEADER = "## Additional Rules (automated patch)"


def build_patched_prompt(
    failed_agent: AgentName,
    patch: str,
    current_prompts: dict[str, str],
) -> dict[str, str]:
    """Return a new prompts dict with only failed_agent's prompt patched."""
    if failed_agent not in current_prompts:
        raise ValueError(f"unknown failed_agent: {failed_agent}")
    patched = dict(current_prompts)
    base = current_prompts[failed_agent].rstrip()
    patched[failed_agent] = f"{base}\n\n{PATCH_HEADER}\n{patch.strip()}\n"
    return patched


def _capture_to_stage_io(cap_stage: Any) -> StageIO | None:
    if cap_stage is None:
        return None
    if isinstance(cap_stage, dict):
        return StageIO(
            input=cap_stage.get("input") or {},
            output=cap_stage.get("output") or {},
            trajectory=cap_stage.get("trajectory") or [],
        )
    return None


def _turns_for_report(report_id: str) -> list[dict[str, Any]]:
    """Return the per-turn rows for a report_id, sorted by q_order/turn_index."""
    group = qa_data[qa_data["report_id"] == report_id].sort_values("q_order")
    return [
        {
            "turn_index": int(idx),
            "question": str(row["conv_questions"]),
            "gold_answer": str(row["conv_answers"]),
        }
        for idx, (_, row) in enumerate(group.iterrows())
    ]


async def verify_patch(
    failed_agent: AgentName,
    patch: str,
    *,
    iteration: int,
    report_id: str,
    failed_turn_index: int,
    current_prompts: dict[str, str],
) -> FixAttempt:
    """Patch failed_agent's prompt, replay turns 0..k, return a FixAttempt.

    Pass iff turn k matches its gold AND no prior turn 0..k-1 regresses.
    """
    patched_prompts = build_patched_prompt(failed_agent, patch, current_prompts)
    agents = make_agents(patched_prompts)
    conversation = ConversationHistory()

    turns = _turns_for_report(report_id)
    if not turns:
        raise RuntimeError(f"no turns found for report_id={report_id}")
    k = failed_turn_index

    turn_results: list[TurnResult] = []
    first_failing: int | None = None
    last_capture: dict[str, Any] | None = None
    last_pred_answer: str = ""

    for t in turns:
        if t["turn_index"] > k:
            break
        capture: dict[str, Any] = {}
        try:
            pred_answer, _ = await run_turn(
                t["question"], report_id, conversation, agents=agents, capture=capture
            )
        except Exception as exc:  # noqa: BLE001
            pred_answer = ""
            capture = {"error": repr(exc)}
        last_capture = capture
        last_pred_answer = str(pred_answer)
        correct = numeric_match(pred_answer, t["gold_answer"])
        turn_results.append(
            TurnResult(
                turn_index=t["turn_index"],
                question=t["question"],
                gold_answer=t["gold_answer"],
                pred_answer=last_pred_answer,
                correct=correct,
            )
        )
        if not correct and first_failing is None:
            first_failing = t["turn_index"]
            break

    passed = first_failing is None and any(tr.turn_index == k for tr in turn_results)
    failure_reason: FailureReason | None = None
    if not passed:
        if first_failing is None:
            failure_reason = "did_not_fix"
        elif first_failing < k:
            failure_reason = "caused_regression"
        else:
            failure_reason = "did_not_fix"

    cap = last_capture or {}

    return FixAttempt(
        iteration=iteration,
        failed_agent=failed_agent,
        patch_applied=patch,
        full_prompt=patched_prompts[failed_agent],
        turn_results=turn_results,
        correct=passed,
        first_failing_turn=first_failing,
        triage_io=_capture_to_stage_io(cap.get("triage")),
        preprocess_io=_capture_to_stage_io(cap.get("preprocess")),
        retriever_io=_capture_to_stage_io(cap.get("retriever")),
        calculator_io=_capture_to_stage_io(cap.get("calculator")),
        verify_result="passed" if passed else "failed",
        failure_reason=failure_reason,
    )
