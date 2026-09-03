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
    """One gold question of the eval-loop dataset, for human review.

    The four ``expected_*`` fields are the per-subagent gold, *derived* from the
    gold program and gold answer rather than labelled: what triage should have
    classified this turn as, the operation skeleton preprocess should have
    planned, the document values the retriever was responsible for finding, and
    what the calculator should have produced. They are the same derivation the
    attribution rule uses, shown so a human can check the rule rather than
    take it on trust — which is where a disputed attribution gets settled.
    """

    split: str
    report_id: str
    turn_index: int
    question: str
    gold_answer: str
    gold_program: str
    turn_type: str
    conv_type: str
    expected_triage: str = ""
    expected_skeleton: list[str] = []
    expected_operands: list[str] = []
    expected_answer: str = ""


class LoopRunSummary(BaseModel):
    """One committed eval-loop run: a split × version pass and what it scored.

    The loop's evidence lives under ``evaluation/predictions/evalloop/`` with its
    own denominator per run, never the 770-question corpus, so ``n_questions``
    is the run's own count and ``accuracy`` is over exactly those rows.
    """

    version: str
    composition: str | None
    split: str
    n_reports: int
    n_questions: int
    n_correct: int
    accuracy: float
    file: str


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


class CampaignExperiment(BaseModel):
    """One gated experiment: what it changed, and what the gate said.

    Shaped from `story.json` rather than queried live, so the Experiments tab
    shows exactly what the published page shows — a discrepancy between the two
    would be a discrepancy about the same tracking store, which is worse than
    having only one of them.
    """

    label: str
    campaign: str
    target_agent: str
    baseline_version: str
    candidate_version: str
    promoted: bool
    at: int | None = None
    accuracy_delta: float | None = None
    cluster_p_one_sided: float | None = None
    delta_ci_lo: float | None = None
    delta_ci_hi: float | None = None
    n_compared: float | None = None
    fixed: float | None = None
    broken: float | None = None
    accuracy_baseline: float | None = None
    accuracy_candidate: float | None = None
    panel_baseline: dict[str, float | None] = {}
    panel_candidate: dict[str, float | None] = {}
    summary_of_changes: str = ""
    rationale: str = ""
    diff: str = ""


class CampaignSummary(BaseModel):
    """A campaign as a unit of review: its experiments and how many promoted."""

    name: str
    n_experiments: int
    n_promoted: int
    n_remaining: int
    blocked_agents: list[str] = []
    complete: bool = False


class ChampionPoint(BaseModel):
    """One point on the champion track: accuracy and the per-agent panel."""

    version: str
    at: int | None = None
    accuracy: float | None = None
    panel: dict[str, float | None] = {}
    moved_by: str | None = None
    target_agent: str | None = None


class CampaignsResponse(BaseModel):
    """Everything the campaign views need, in one request.

    `champion_accuracy` is the champion's own accuracy on the fixed gate split —
    the figure the campaign actually optimises and gates against, and the one
    the status board leads with. It is deliberately not the legacy 770-question
    corpus number: that corpus is a different population, scored by a different
    protocol, and its newest committed version is one the campaign rolled back.
    """

    champion: str | None = None
    champion_accuracy: float | None = None
    champion_panel: dict[str, float | None] = {}
    rule: str = ""
    generated_at: str = ""
    split: dict[str, Any] = {}
    campaigns: list[CampaignSummary] = []
    experiments: list[CampaignExperiment] = []
    champion_track: list[ChampionPoint] = []
