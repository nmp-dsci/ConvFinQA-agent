"""FastAPI app: session-backed chat + eval-runs endpoints for the frontend."""

# ruff: noqa: D102, D103

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import logfire
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from convfinqa.config import PREDICTIONS_DIR, settings
from convfinqa.data.loader import _DOCS, qa_data
from convfinqa.data.schemas import ConversationHistory
from convfinqa.pipeline.runner import run_turn, stream_turn

REPORT_IDS = sorted(set(qa_data["report_id"]).intersection(_DOCS))
QUESTIONS_DF = qa_data[qa_data["report_id"].isin(REPORT_IDS)].copy()

EVAL_DIR = PREDICTIONS_DIR

_MODEL_CSV_PATTERN: dict[str, str] = {
    "dspy": "dspy_predictions_{v}_joined.csv",
    "pydantic": "pydantic_predictions_{v}_joined.csv",
    "api": "api_predictions_{v}_joined.csv",
}
_GOLD_PROGRAMS: dict[tuple[str, int], str] = {
    (str(r.report_id), int(r.q_order)): str(r.turn_program)
    for r in qa_data.itertuples()
}


def _eval_version_key(v: str) -> tuple[int, int]:
    """Sort key for version labels: ``v1`` → (1, 0), ``v3_1`` → (3, 1).

    Always returns a uniform ``(int, int)`` so mixed plain/variant versions
    (``v1``, ``v2``, ``v3_1``) order without comparing across types.
    Unparseable labels sort last.
    """
    body = v[1:] if v.startswith("v") else v
    parts = body.split("_")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (10_000, 0)
    return (major, minor)


class ReportSummary(BaseModel):
    report_id: str
    n_questions: int
    doc_summary: str


class ReportDocument(BaseModel):
    report_id: str
    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | str | int]]


class ReportQuestion(BaseModel):
    q_order: int
    question: str
    gold_answer: str
    gold_program: str


class CreateSessionRequest(BaseModel):
    report_id: str


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class HistoryItem(BaseModel):
    question: str
    answer: str
    report_id: str


class SessionResponse(BaseModel):
    session_id: str
    report_id: str
    created_at: datetime
    updated_at: datetime
    n_turns: int
    history: list[HistoryItem]


class AskResponse(BaseModel):
    answer: str
    turn_index: int
    history: list[HistoryItem]


class AccuracySlice(BaseModel):
    label: str
    accuracy: float
    n_correct: int
    n_total: int


class ModelAccuracy(BaseModel):
    overall: AccuracySlice
    by_turn_type: list[AccuracySlice]
    by_conv_type: list[AccuracySlice]
    by_q_order: list[AccuracySlice]


class EvalSummary(BaseModel):
    run_name: str
    available_models: list[str]
    models: dict[str, ModelAccuracy]


class PredRow(BaseModel):
    report_id: str
    turn_index: int
    question: str
    gold_answer: str
    gold_program: str
    pred_answer: str
    pred_program: str
    correct: bool
    q_order: int
    turn_type: str
    conv_type: str


@dataclass
class SessionState:
    session_id: str
    report_id: str
    created_at: datetime
    updated_at: datetime
    conversation: ConversationHistory = field(default_factory=ConversationHistory)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def as_response(self) -> SessionResponse:
        return SessionResponse(
            session_id=self.session_id,
            report_id=self.report_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            n_turns=len(self.conversation.pairs),
            history=[
                HistoryItem.model_validate(p.model_dump())
                for p in self.conversation.pairs
            ],
        )


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, SessionState] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def create(self, report_id: str) -> SessionState:
        if report_id not in REPORT_IDS:
            raise KeyError(report_id)
        now = datetime.now(timezone.utc)
        state = SessionState(
            session_id=str(uuid4()),
            report_id=report_id,
            created_at=now,
            updated_at=now,
        )
        self.sessions[state.session_id] = state
        self.locks[state.session_id] = asyncio.Lock()
        return state

    def get(self, session_id: str) -> SessionState:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.locks.pop(session_id, None)

    def get_lock(self, session_id: str) -> asyncio.Lock:
        try:
            return self.locks[session_id]
        except KeyError as exc:
            raise KeyError(session_id) from exc

    def evict_expired(self) -> list[str]:
        now = datetime.now(timezone.utc)
        expired = [
            sid
            for sid, state in self.sessions.items()
            if (now - state.updated_at).total_seconds() > self.ttl_seconds
        ]
        for sid in expired:
            self.delete(sid)
        return expired


def _history_items(conversation: ConversationHistory) -> list[HistoryItem]:
    return [HistoryItem.model_validate(p.model_dump()) for p in conversation.pairs]


def _report_questions(report_id: str) -> list[ReportQuestion]:
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
    n_questions = len(_report_questions(report_id))
    summary = f"pre_text: {pre} | post_text: {post}".strip()
    return ReportSummary(
        report_id=report_id,
        n_questions=n_questions,
        doc_summary=summary[:400],
    )


def _load_preds(version: str, model: str) -> pd.DataFrame | None:
    path = EVAL_DIR / _MODEL_CSV_PATTERN[model].format(v=version)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(str).str.lower() == "true"
    df["q_order"] = df["q_order"].astype(float).astype(int)
    if "pred_program" not in df.columns:
        df["pred_program"] = ""
    return df


def _slice_accuracy(df: pd.DataFrame, label: str) -> AccuracySlice:
    n = len(df)
    c = int(df["correct"].sum())
    return AccuracySlice(
        label=label, accuracy=round(c / n, 4) if n else 0.0, n_correct=c, n_total=n
    )


def _slices_by(df: pd.DataFrame, col: str) -> list[AccuracySlice]:
    result = []
    for val in sorted(df[col].dropna().unique(), key=str):
        result.append(_slice_accuracy(df[df[col] == val], str(val)))
    return result


def create_app(
    *,
    session_ttl_seconds: int = 1800,
    eviction_interval_seconds: int = 60,
) -> FastAPI:
    store = SessionStore(ttl_seconds=session_ttl_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        eviction_task = asyncio.create_task(
            _eviction_loop(store, eviction_interval_seconds)
        )
        try:
            yield
        finally:
            eviction_task.cancel()
            with suppress(asyncio.CancelledError):
                await eviction_task

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins.split(","),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.session_store = store
    app.state.session_ttl_seconds = session_ttl_seconds
    app.state.eviction_interval_seconds = eviction_interval_seconds

    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()
    logfire.instrument_fastapi(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/reports")
    async def list_reports(
        q: str = "",
        limit: int = Query(default=20, ge=1, le=500),
    ) -> list[str]:
        needle = q.lower()
        matches = [rid for rid in REPORT_IDS if needle in rid.lower()]
        return matches[:limit]

    @app.get("/reports/{report_id:path}/questions")
    async def get_report_questions(report_id: str) -> list[ReportQuestion]:
        _ensure_report_exists(report_id)
        return _report_questions(report_id)

    @app.get("/reports/{report_id:path}/document")
    async def get_report_document(report_id: str) -> ReportDocument:
        _ensure_report_exists(report_id)
        doc = _DOCS[report_id]
        return ReportDocument(
            report_id=report_id,
            pre_text=doc.pre_text,
            post_text=doc.post_text,
            table=doc.table,
        )

    @app.get("/reports/{report_id:path}")
    async def get_report(report_id: str) -> ReportSummary:
        _ensure_report_exists(report_id)
        return _report_summary(report_id)

    @app.post("/sessions")
    async def create_session(body: CreateSessionRequest) -> SessionResponse:
        try:
            state = store.create(body.report_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown report_id: {body.report_id}"
            ) from exc
        return state.as_response()

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> SessionResponse:
        state = _get_session_or_404(store, session_id)
        return state.as_response()

    @app.post("/sessions/{session_id}/ask")
    async def ask(session_id: str, body: AskRequest) -> AskResponse:
        lock = _get_lock_or_404(store, session_id)
        async with lock:
            state = _get_session_or_404(store, session_id)
            result = await run_turn(body.question, state.report_id, state.conversation)
            answer = result[0] if isinstance(result, tuple) else result
            state.touch()
            return AskResponse(
                answer=answer,
                turn_index=len(state.conversation.pairs) - 1,
                history=_history_items(state.conversation),
            )

    @app.post("/sessions/{session_id}/ask/stream")
    async def ask_stream(session_id: str, body: AskRequest) -> StreamingResponse:
        lock = _get_lock_or_404(store, session_id)

        async def gen() -> AsyncIterator[str]:
            async with lock:
                state = _get_session_or_404(store, session_id)
                turn_index = len(state.conversation.pairs)
                with logfire.span(
                    "ask {report_id} turn={turn_index}",
                    report_id=state.report_id,
                    session_id=session_id,
                    question=body.question,
                    turn_index=turn_index,
                ):
                    try:
                        async for event in stream_turn(
                            body.question, state.report_id, state.conversation
                        ):
                            if event.get("event") == "answer":
                                logfire.info("answer", answer=event.get("answer"))
                            yield f"data: {json.dumps(event)}\n\n"
                        state.touch()
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "event": "done",
                                    "turn_index": len(state.conversation.pairs) - 1,
                                }
                            )
                            + "\n\n"
                        )
                    except Exception as exc:  # noqa: BLE001
                        logfire.error("stream error", error=str(exc))
                        yield (
                            "data: "
                            + json.dumps({"event": "error", "error": str(exc)})
                            + "\n\n"
                        )

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        _get_session_or_404(store, session_id)
        store.delete(session_id)
        return Response(status_code=204)

    @app.get("/eval/runs")
    async def list_eval_runs() -> list[str]:
        """Return the prompt versions that have at least one joined CSV in evaluation/."""
        if not EVAL_DIR.exists():
            return []
        versions: set[str] = set()
        for path in EVAL_DIR.iterdir():
            if not path.is_file() or path.suffix != ".csv":
                continue
            stem = path.stem
            if not stem.endswith("_joined"):
                continue
            base = stem[: -len("_joined")]
            for model in _MODEL_CSV_PATTERN:
                prefix = f"{model}_predictions_"
                if base.startswith(prefix):
                    versions.add(base[len(prefix) :])
                    break
        return sorted(versions, key=_eval_version_key)

    @app.get("/eval/runs/{run_name}/summary")
    async def get_eval_summary(run_name: str) -> EvalSummary:
        available: dict[str, ModelAccuracy] = {}
        for model in _MODEL_CSV_PATTERN:
            df = _load_preds(run_name, model)
            if df is None:
                continue
            available[model] = ModelAccuracy(
                overall=_slice_accuracy(df, "overall"),
                by_turn_type=_slices_by(df, "turn_type"),
                by_conv_type=_slices_by(df, "conv_type"),
                by_q_order=_slices_by(df, "q_order"),
            )
        if not available:
            raise HTTPException(
                status_code=404, detail=f"No predictions found for version {run_name}"
            )
        return EvalSummary(
            run_name=run_name, available_models=list(available), models=available
        )

    @app.get("/eval/runs/{run_name}/predictions")
    async def get_eval_predictions(
        run_name: str, model: str = "pydantic"
    ) -> list[PredRow]:
        if model not in _MODEL_CSV_PATTERN:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        df = _load_preds(run_name, model)
        if df is None:
            raise HTTPException(
                status_code=404, detail=f"No predictions for {run_name}/{model}"
            )
        rows: list[PredRow] = []
        for row in df.itertuples():
            key = (str(row.report_id), int(row.q_order))
            rows.append(
                PredRow(
                    report_id=str(row.report_id),
                    turn_index=int(row.turn_index),
                    question=str(row.question),
                    gold_answer=str(row.gold_answer),
                    gold_program=_GOLD_PROGRAMS.get(key, ""),
                    pred_answer=str(row.pred_answer),
                    pred_program=str(getattr(row, "pred_program", "") or ""),
                    correct=bool(row.correct),
                    q_order=int(row.q_order),
                    turn_type=str(row.turn_type),
                    conv_type=str(row.conv_type),
                )
            )
        return rows

    return app


async def _eviction_loop(store: SessionStore, interval_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        store.evict_expired()


def _ensure_report_exists(report_id: str) -> None:
    if report_id not in REPORT_IDS:
        raise HTTPException(status_code=404, detail=f"Unknown report_id: {report_id}")


def _get_session_or_404(store: SessionStore, session_id: str) -> SessionState:
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown session_id: {session_id}"
        ) from exc


def _get_lock_or_404(store: SessionStore, session_id: str) -> asyncio.Lock:
    try:
        return store.get_lock(session_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown session_id: {session_id}"
        ) from exc


app = create_app()
