"""Reports, sessions, and the ask/stream endpoints.

The one structural decision worth naming: demo mode branches *here*, at the point
where a turn is produced, and nowhere else. Both branches emit the same event
stream and both append to the same conversation history, so every other part of
the system — the trace store, the frontend, the session model — is unaware that
two paths exist.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import logfire
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from convfinqa.data.loader import _DOCS, qa_data
from convfinqa.error_codes import classify
from convfinqa.llm import DemoModeError, demo_mode_enabled
from convfinqa.serving import evaldata
from convfinqa.serving.demo_pack import replay
from convfinqa.serving.limits import client_key
from convfinqa.serving.models import (
    AskRequest,
    AskResponse,
    CreateSessionRequest,
    DemoQuestion,
    ReportDocument,
    ReportQuestion,
    ReportSummary,
    SessionResponse,
)
from convfinqa.serving.sessions import SessionState, SessionStore, history_items

router = APIRouter()

REPORT_IDS: list[str] = sorted(set(qa_data["report_id"]).intersection(_DOCS))
QUESTIONS_DF = qa_data[qa_data["report_id"].isin(REPORT_IDS)].copy()


def _store(request: Request) -> SessionStore:
    store: SessionStore = request.app.state.session_store
    return store


def _session_or_404(store: SessionStore, session_id: str) -> SessionState:
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown session_id: {session_id}"
        ) from exc


def _lock_or_404(store: SessionStore, session_id: str) -> Any:
    try:
        return store.get_lock(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown session_id: {session_id}"
        ) from exc


def _ensure_report(report_id: str) -> None:
    if report_id not in REPORT_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")


def report_questions(report_id: str) -> list[ReportQuestion]:
    """Gold questions for one report, in conversation order."""
    df = QUESTIONS_DF[QUESTIONS_DF["report_id"] == report_id].sort_values("q_order")
    return [
        ReportQuestion(
            q_order=int(row.q_order),
            question=str(row.conv_questions),
            gold_answer=str(row.conv_answers),
            gold_program=str(row.turn_program),
        )
        for row in df.itertuples()
    ]


def _report_summary(report_id: str) -> ReportSummary:
    doc = _DOCS[report_id]
    pre = " ".join(doc.pre_text.split()[:24])
    post = " ".join(doc.post_text.split()[:24])
    summary = f"pre_text: {pre} | post_text: {post}".strip()
    packed = {entry["report_id"] for entry in replay.packed_reports()}
    return ReportSummary(
        report_id=report_id,
        n_questions=len(report_questions(report_id)),
        doc_summary=summary[:400],
        split=evaldata.split_of().get(report_id, "unknown"),
        in_demo_pack=report_id in packed,
    )


@router.get("/reports")
async def list_reports(
    q: str = "",
    limit: int = Query(default=20, ge=1, le=500),
    demo_only: bool = False,
) -> list[str]:
    """Report ids, optionally filtered to those the demo can converse about."""
    needle = q.lower()
    pool = REPORT_IDS
    if demo_only or demo_mode_enabled():
        packed = [entry["report_id"] for entry in replay.packed_reports()]
        if packed:
            pool = packed
    return [rid for rid in pool if needle in rid.lower()][:limit]


@router.get("/reports/{report_id:path}/questions")
async def get_report_questions(report_id: str) -> list[ReportQuestion]:
    """Gold questions for a report."""
    _ensure_report(report_id)
    return report_questions(report_id)


@router.get("/reports/{report_id:path}/document")
async def get_report_document(report_id: str) -> ReportDocument:
    """The financial document behind a report."""
    _ensure_report(report_id)
    doc = _DOCS[report_id]
    return ReportDocument(
        report_id=report_id,
        pre_text=doc.pre_text,
        post_text=doc.post_text,
        table=doc.table,
    )


@router.get("/reports/{report_id:path}")
async def get_report(report_id: str) -> ReportSummary:
    """Summary of one report, including its split and demo availability."""
    _ensure_report(report_id)
    return _report_summary(report_id)


@router.get("/demo/reports")
async def demo_reports() -> list[dict[str, Any]]:
    """Reports the recorded pack can hold a conversation about."""
    return replay.packed_reports()


@router.get("/demo/questions")
async def demo_questions(report_id: str) -> list[DemoQuestion]:
    """The chip rail: questions this report can actually answer in demo mode."""
    return [DemoQuestion(**entry) for entry in replay.suggested_questions(report_id)]


@router.post("/sessions")
async def create_session(
    body: CreateSessionRequest, request: Request
) -> SessionResponse:
    """Open a session against a report."""
    try:
        state = _store(request).create(body.report_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown report_id: {body.report_id}"
        ) from exc
    return state.as_response()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> SessionResponse:
    """Fetch a session and its history."""
    return _session_or_404(_store(request), session_id).as_response()


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> Response:
    """Close a session."""
    store = _store(request)
    _session_or_404(store, session_id)
    store.delete(session_id)
    return Response(status_code=204)


async def _turn_stream(
    state: SessionState,
    question: str,
    capture: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """The one place demo and live diverge. Both yield the same event vocabulary."""
    if demo_mode_enabled():
        # `capture` is threaded through so a replayed turn lands in the trace
        # store with the stage metrics it was recorded with, rather than as a
        # turn that apparently cost nothing.
        async for event in replay.replay_turn(
            question, state.report_id, state.conversation, capture=capture
        ):
            yield event
        return
    from convfinqa.pipeline.runner import turn_events

    async for event in turn_events(
        question, state.report_id, state.conversation, capture=capture
    ):
        yield event


def _record_trace(
    state: SessionState,
    question: str,
    capture: dict[str, Any],
    answer: str,
    program: str,
    turn_index: int,
    error: str = "",
    error_code: str = "",
) -> str:
    """Persist the turn to the trace store. Never raises into the request."""
    from convfinqa.tracking.traces import get_store

    store = get_store()
    if store is None:
        return ""
    gold = {q.q_order: q.gold_answer for q in report_questions(state.report_id)}.get(
        turn_index
    )
    correct: bool | None = None
    if gold is not None and answer:
        from convfinqa.evaluation import numeric_match

        correct = bool(numeric_match(answer, gold))
    return store.record(
        report_id=state.report_id,
        turn_index=turn_index,
        question=question,
        capture=capture,
        answer=answer,
        program=program,
        source="demo" if demo_mode_enabled() else "serving",
        session_id=state.session_id,
        gold_answer=gold,
        correct=correct,
        error=error,
        error_code=error_code,
    )


@router.post("/sessions/{session_id}/ask")
async def ask(session_id: str, body: AskRequest, request: Request) -> AskResponse:
    """Answer one question, returning the final answer only."""
    store = _store(request)
    lock = _lock_or_404(store, session_id)
    limiter = request.app.state.inflight
    await limiter.acquire()
    try:
        async with lock:
            state = _session_or_404(store, session_id)
            turn_index = len(state.conversation.pairs)
            capture: dict[str, Any] = {}
            answer = ""
            program = ""
            matched_question = ""
            try:
                async for event in _turn_stream(state, body.question, capture):
                    if event.get("event") == "answer":
                        answer = str(event.get("answer", ""))
                        program = str(event.get("program", ""))
                    elif event.get("event") == "matched":
                        matched_question = str(event.get("matched_question", ""))
            except DemoModeError as exc:
                _record_trace(
                    state,
                    body.question,
                    capture,
                    "",
                    "",
                    turn_index,
                    error=str(exc),
                    error_code=classify(exc),
                )
                raise HTTPException(
                    status_code=501, detail={"code": exc.code, "message": str(exc)}
                ) from exc
            except replay.NoRecordingError as exc:
                _record_trace(
                    state,
                    body.question,
                    capture,
                    "",
                    "",
                    turn_index,
                    error=str(exc),
                    error_code=classify(exc),
                )
                raise HTTPException(
                    status_code=404, detail={"code": exc.code, "message": str(exc)}
                ) from exc
            state.touch()
            trace_id = _record_trace(
                state, body.question, capture, answer, program, turn_index
            )
            return AskResponse(
                answer=answer,
                turn_index=len(state.conversation.pairs) - 1,
                history=history_items(state.conversation),
                trace_id=trace_id,
                matched_question=matched_question,
            )
    finally:
        await limiter.release()


@router.post("/sessions/{session_id}/ask/stream")
async def ask_stream(
    session_id: str, body: AskRequest, request: Request
) -> StreamingResponse:
    """Answer one question, streaming each stage as it completes."""
    store = _store(request)
    lock = _lock_or_404(store, session_id)
    limiter = request.app.state.inflight
    await limiter.acquire()

    async def gen() -> AsyncIterator[str]:
        try:
            async with lock:
                state = _session_or_404(store, session_id)
                turn_index = len(state.conversation.pairs)
                capture: dict[str, Any] = {}
                answer = ""
                program = ""
                matched_question = ""
                with logfire.span(
                    "ask {report_id} turn={turn_index}",
                    report_id=state.report_id,
                    session_id=session_id,
                    question=body.question,
                    turn_index=turn_index,
                ):
                    try:
                        async for event in _turn_stream(state, body.question, capture):
                            if event.get("event") == "answer":
                                answer = str(event.get("answer", ""))
                                program = str(event.get("program", ""))
                                logfire.info("answer", answer=answer)
                            elif event.get("event") == "matched":
                                matched_question = str(
                                    event.get("matched_question", "")
                                )
                            yield f"data: {json.dumps(event)}\n\n"
                        state.touch()
                        trace_id = _record_trace(
                            state, body.question, capture, answer, program, turn_index
                        )
                        yield _frame(
                            {
                                "event": "done",
                                "turn_index": len(state.conversation.pairs) - 1,
                                "trace_id": trace_id,
                                "matched_question": matched_question,
                            }
                        )
                    except (DemoModeError, replay.NoRecordingError) as exc:
                        # Typed, actionable refusals — the frontend maps `code`
                        # to its own copy rather than showing a raw message.
                        code = classify(exc)
                        _record_trace(
                            state,
                            body.question,
                            capture,
                            answer,
                            program,
                            turn_index,
                            error=str(exc),
                            error_code=code,
                        )
                        yield _frame(
                            {"event": "error", "code": code, "error": str(exc)}
                        )
                    except Exception as exc:  # noqa: BLE001
                        logfire.error("stream error", error=str(exc))
                        code = classify(exc)
                        _record_trace(
                            state,
                            body.question,
                            capture,
                            answer,
                            program,
                            turn_index,
                            error=str(exc),
                            error_code=code,
                        )
                        yield _frame(
                            {"event": "error", "code": code, "error": str(exc)}
                        )
        finally:
            await limiter.release()

    return StreamingResponse(gen(), media_type="text/event-stream")


def _frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


def rate_limit_key(request: Request) -> str:
    """Re-exported for the middleware, which lives in the app factory."""
    return client_key(request)
