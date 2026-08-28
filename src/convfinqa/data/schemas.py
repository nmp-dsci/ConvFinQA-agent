"""Shared data schemas for ConvFinQA pipelines."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TurnType = Literal["number", "program"]
ConvType = Literal["Type I", "Type II"]


class ConvExample(BaseModel):
    report_id: str
    questions: list[str]
    gold_answers: list[str]
    gold_programs: list[str] = Field(default_factory=list)
    gold_turn_types: list[str] = Field(default_factory=list)
    gold_conv_types: list[str] = Field(default_factory=list)


class QAPair(BaseModel):
    """A question paired with its answer."""

    question: str
    answer: str


class HistoryTurn(BaseModel):
    """One turn of conversation history, tagged with the report it answered against."""

    question: str
    answer: str
    report_id: str


class ConversationHistory(BaseModel):
    """Multi-turn history for a session. May span multiple documents."""

    pairs: list[HistoryTurn] = Field(default_factory=list)

    def append(self, question: str, answer: str, report_id: str) -> None:
        """Append a question/answer/report_id triple to the history."""
        self.pairs.append(
            HistoryTurn(question=question, answer=answer, report_id=report_id)
        )

    def as_text(self) -> str:
        """Format the history as a flat text block for inclusion in agent prompts."""
        if not self.pairs:
            return "(no prior turns)"
        return "\n".join(
            f"Q{i + 1} [report={p.report_id}]: {p.question}\nA{i + 1}: {p.answer}"
            for i, p in enumerate(self.pairs)
        )


class AgentResponse(BaseModel):
    """Final response surfaced to the evaluator/caller."""

    question: str
    report_id: str
    answer: str
    turn_type: TurnType
    conv_type: ConvType
    turn_program: str | None = None
    triage_reasoning: str | None = None
    preprocess_reasoning: str | None = None
    retriever_reasoning: str | None = None
    calc_trajectory: dict[str, Any] | None = None


class Document(BaseModel):
    """Financial document: pre/post text plus structured table."""

    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | str | int]]


class Dialogue(BaseModel):
    """Multi-turn dialogue with gold programs and executed answers."""

    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | str]
    qa_split: list[bool] = Field(default_factory=list)


class Features(BaseModel):
    """Helper features computed from the dialogue."""

    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    """One record from the ConvFinQA dataset."""

    id: str
    doc: Document
    dialogue: Dialogue
    features: Features
