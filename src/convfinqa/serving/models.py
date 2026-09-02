"""Response models shared across the serving routers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """What `/healthz` reports. The frontend reads `mode` to configure itself.

    One build serves both deployments; this payload is how it finds out which
    one it is running in, so there is never a demo-specific bundle to keep in
    sync with the real one.
    """

    ok: bool
    mode: str
    champion: str | None
    bundle_id: str
    bundle: dict[str, Any]
    demo_reports: int


class ReportSummary(BaseModel):
    report_id: str
    n_questions: int
    doc_summary: str
    split: str = "unknown"
    in_demo_pack: bool = False


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
    trace_id: str = ""
    # Set only in demo mode, and only when the asked question was resolved to a
    # *different* recorded one. The UI owes the visitor a banner when it is set:
    # answering a paraphrase silently would present one question's number as
    # another's.
    matched_question: str = ""


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


class DatasetRow(BaseModel):
    """One gold question of the eval-loop dataset, for human review."""

    split: str
    report_id: str
    turn_index: int
    question: str
    gold_answer: str
    gold_program: str
    turn_type: str
    conv_type: str


class SplitSummary(BaseModel):
    """Dataset split membership, made visible rather than merely claimed."""

    name: str
    description: str
    n_conversations: int
    n_questions: int
    report_ids: list[str]


class VersionAnswer(BaseModel):
    """One version's answer to one question, for the side-by-side explorer."""

    version: str
    pred_answer: str
    pred_program: str
    correct: bool


class AnswerRow(BaseModel):
    """A question with gold plus every version's answer beside it."""

    report_id: str
    turn_index: int
    question: str
    gold_answer: str
    gold_program: str
    gold_turn_type: str
    gold_conv_type: str
    versions: list[VersionAnswer]


class TraceSummary(BaseModel):
    trace_id: str
    created_at: str
    source: str
    session_id: str | None
    report_id: str
    turn_index: int
    question: str
    answer: str | None
    program: str | None
    gold_answer: str | None
    correct: bool | None
    bundle_id: str | None
    latency_ms: float | None
    total_tokens: int | None
    cost_usd: float | None = None
    error: str | None
    error_code: str | None = None


class DemoQuestion(BaseModel):
    turn_index: int
    question: str
    gold_answer: str
    correct: bool
