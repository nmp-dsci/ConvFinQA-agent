"""Pydantic data models for the s7 diagnosis harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FailedAgent = Literal["triage", "preprocess", "retriever", "calculator", "ambiguous"]
FixType = Literal["add_rule", "modify_rule", "add_example", "clarify_instruction"]
AgentName = Literal["triage", "preprocess", "retriever", "calculator"]
VerifyResult = Literal["passed", "failed"]
FailureReason = Literal[
    "did_not_fix",
    "caused_regression",
    "duplicate_patch",
    "ambiguous_followup",
]


class StageIO(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)


class TurnResult(BaseModel):
    turn_index: int
    question: str
    gold_answer: str
    pred_answer: str
    correct: bool


class FixAttempt(BaseModel):
    iteration: int
    failed_agent: str
    patch_applied: str
    full_prompt: str
    # Step 2 propose metadata — populated when a propose was made; reused by
    # the Step 2 cache to skip the specialist LLM call. Optional for back-compat
    # with case_results_<variant>.jsonl lines written before this field existed.
    fix_type: FixType | None = None
    fix_confidence: float | None = None
    turn_results: list[TurnResult] = Field(default_factory=list)
    correct: bool = False
    first_failing_turn: int | None = None
    triage_io: StageIO | None = None
    preprocess_io: StageIO | None = None
    retriever_io: StageIO | None = None
    calculator_io: StageIO | None = None
    verify_result: VerifyResult | None = None
    failure_reason: FailureReason | None = None

    @property
    def pred_answer(self) -> str:
        if not self.turn_results:
            return ""
        idx = (
            self.first_failing_turn
            if self.first_failing_turn is not None
            else self.turn_results[-1].turn_index
        )
        return next(
            (t.pred_answer for t in self.turn_results if t.turn_index == idx),
            "",
        )


class RouterDiagnosis(BaseModel):
    failed_agent: FailedAgent
    failure_mode: str
    failure_explanation: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FixProposal(BaseModel):
    rule: str
    fix_type: FixType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class RuleAttempt(BaseModel):
    attempt_id: str
    agent: AgentName
    rule: str
    fix_type: FixType
    confidence: float
    verify_result: VerifyResult
    attempted_on: dict[str, Any]
    attempted_at: str
    first_failing_turn: int | None = None
    failure_reason: FailureReason | None = None
    promoted_rule_id: str | None = None


class Rule(BaseModel):
    rule_id: str
    agent: AgentName
    rule: str
    fix_type: FixType
    confidence: float
    verified_on: list[dict[str, Any]] = Field(default_factory=list)
    verified_at: str
    supersedes: list[str] = Field(default_factory=list)


class RouterPayload(BaseModel):
    """Input to diagnostic_router_agent (Step 1 — Diagnose)."""

    report_id: str
    turn_index: int
    question: str
    history_text: str
    gold_answer: str
    pred_answer: str
    gold_program: str
    gold_turn_type: str
    pred_turn_type: str
    gold_conv_type: str
    pred_conv_type: str
    triage_io: StageIO | None = None
    preprocess_io: StageIO | None = None
    retriever_io: StageIO | None = None
    calculator_io: StageIO | None = None
    current_triage_prompt: str
    current_preprocess_prompt: str
    current_retriever_prompt: str
    current_calculator_prompt: str


class FixPayload(BaseModel):
    """Input to specialist fix agent (Step 2 — Route+Fix)."""

    report_id: str
    turn_index: int
    question: str
    history_text: str
    gold_answer: str
    pred_answer: str
    gold_program: str
    router_diagnosis: RouterDiagnosis
    failed_agent_io: StageIO | None = None
    upstream_ios: dict[str, StageIO | None] = Field(default_factory=dict)
    current_prompt: str
    prior_rule_attempts: list[RuleAttempt] = Field(default_factory=list)
    prior_attempts: list[FixAttempt] = Field(default_factory=list)


class CaseResult(BaseModel):
    report_id: str
    turn_index: int
    question: str
    gold_answer: str
    original_pred_answer: str
    gold_turn_type: str
    gold_program: str
    router_diagnosis: RouterDiagnosis | None = None
    attempts: list[FixAttempt] = Field(default_factory=list)
    resolved: bool = False
    winning_iteration: int | None = None
    final_patch: str | None = None
