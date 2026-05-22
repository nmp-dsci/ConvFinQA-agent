"""Pydantic models for pipeline stage outputs."""

from __future__ import annotations

from pydantic import BaseModel

from convfinqa.data.schemas import ConvType, QAPair, TurnType


class TriageOut(BaseModel):
    """Mirrors TriageSignature outputs."""

    reasoning: str
    turn_type: TurnType
    conv_type: ConvType


class PreprocessOut(BaseModel):
    """Mirrors PreprocessSignature outputs."""

    reasoning: str
    sub_questions: list[str]
    program: str


class RetrievedValues(BaseModel):
    """Mirrors RetrieverSignature outputs."""

    reasoning: str
    answers: list[QAPair]


class CalcOut(BaseModel):
    """Mirrors CalculationSignature output."""

    answer: str
