"""Trace browsing: the in-app answer to "why did it answer that".

Serves both live serving turns and, via `?source=eval`, turns replayed out of the
committed prediction CSVs — the same viewer either way, because both were
produced by the same `capture` structure.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from convfinqa.serving import evaldata
from convfinqa.serving.models import TraceSummary
from convfinqa.tracking.traces import get_store

router = APIRouter(prefix="/traces")


@router.get("")
async def list_traces(
    report_id: str = "",
    session_id: str = "",
    source: str = "",
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[TraceSummary]:
    """Recent traces, newest first."""
    store = get_store()
    if store is None:
        return []
    rows = store.list_turns(
        report_id=report_id or None,
        session_id=session_id or None,
        source=source or None,
        limit=limit,
        offset=offset,
    )
    return [TraceSummary.model_validate(row) for row in rows]


@router.get("/stats")
async def trace_stats() -> dict[str, Any]:
    """Headline counts for the traces tab."""
    store = get_store()
    return store.stats() if store else {"n_turns": 0, "n_reports": 0}


@router.get("/eval/{version}/{report_id:path}")
async def eval_trace(
    version: str, report_id: str, turn_index: int = 0
) -> dict[str, Any]:
    """Reconstruct a stage timeline for one scored eval turn.

    The prediction CSVs carry the same per-stage IO the live path records, so an
    eval turn from a year-old run opens in exactly the viewer a live turn does.
    """
    df = evaldata.load_joined(version)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No predictions for {version}")
    match = df[(df["report_id"] == report_id) & (df["turn_index"] == turn_index)]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No scored turn {turn_index} for {report_id} in {version}",
        )
    row = match.iloc[0]
    capture = {
        stage: _loads(row.get(f"{stage}_io"))
        for stage in ("triage", "preprocess", "retriever", "calculator")
    }
    return {
        "version": version,
        "report_id": report_id,
        "turn_index": turn_index,
        "question": str(row.get("question", "")),
        "answer": str(row.get("pred_answer", "")),
        "program": str(row.get("pred_program", "") or ""),
        "gold_answer": str(row.get("gold_answer", "")),
        "correct": bool(row["correct"]),
        "history_text": str(row.get("history_text", "") or ""),
        "capture": capture,
        "source": "eval",
    }


@router.get("/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """One trace with its full per-stage capture."""
    store = get_store()
    record = store.get_turn(trace_id) if store else None
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown trace_id: {trace_id}")
    return record


def _loads(raw: Any) -> Any:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
