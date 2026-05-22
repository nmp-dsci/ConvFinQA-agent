"""Compatibility layer for the original DSPy agent tests."""

# ruff: noqa: D102,D103

from __future__ import annotations

import json
import random
from typing import Any

import dspy
from pydantic import BaseModel, Field

from convfinqa.data.loader import load_raw_dataset
from convfinqa.data.schemas import ConvFinQARecord, QAPair
from convfinqa.pipeline.tools import (  # noqa: F401
    CALCULATOR_TOOLS,
    add,
    divide,
    exp,
    greater,
    multiply,
    subtract,
)


class AgentResponse(BaseModel):
    answer: str
    turn_type: str
    conv_type: str
    program: str | None = None
    sub_questions: list[QAPair] = Field(default_factory=list)


class _HistoryTurn(BaseModel):
    question: str
    answer: str
    report_id: str = ""


class ConversationHistory(BaseModel):
    pairs: list[_HistoryTurn] = Field(default_factory=list)

    def append(self, question: str, answer: str, report_id: str = "") -> None:
        self.pairs.append(_HistoryTurn(question=question, answer=answer, report_id=report_id))

    def as_text(self) -> str:
        if not self.pairs:
            return "(no prior turns)"
        return "\n".join(
            f"Q{i + 1}: {p.question}\nA{i + 1}: {p.answer}"
            for i, p in enumerate(self.pairs)
        )


def numeric_match(pred: Any, gold: Any) -> bool:
    try:
        return abs(float(pred) - float(gold)) <= 1e-3
    except (TypeError, ValueError):
        return str(pred).strip().lower() == str(gold).strip().lower()


def sample_records(n: int = 2, seed: int = 42) -> list[ConvFinQARecord]:
    records = [ConvFinQARecord.model_validate(r) for r in load_raw_dataset()["train"]]
    rng = random.Random(seed)
    return rng.sample(records, n)


def serialize_document(record: ConvFinQARecord) -> str:
    return (
        f"PRE-TEXT:\n{record.doc.pre_text}\n\n"
        f"TABLE:\n{json.dumps(record.doc.table, indent=2)}\n\n"
        f"POST-TEXT:\n{record.doc.post_text}"
    )


class ConvFinQAOrchestrator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.triage = dspy.Predict("question, history -> turn_type, conv_type")
        self.preprocess = dspy.Predict("question, history, conv_type -> sub_questions, program")
        self.retriever = dspy.Predict("question, document, history -> answer")
        self.calculator = dspy.Predict("question, retrieved, program -> answer")

    def forward(
        self,
        question: str,
        document: str,
        history: ConversationHistory,
    ) -> AgentResponse:
        triage = self.triage(question=question, history=history.as_text())
        turn_type = str(triage.turn_type)
        conv_type = str(triage.conv_type)
        if turn_type == "number":
            ret = self.retriever(
                question=question,
                document=document,
                history=history.as_text(),
            )
            answer = str(ret.answer)
            return AgentResponse(
                answer=answer,
                turn_type=turn_type,
                conv_type=conv_type,
                sub_questions=[QAPair(question=question, answer=answer)],
            )

        pp = self.preprocess(question=question, history=history.as_text(), conv_type=conv_type)
        retrieved: list[QAPair] = []
        for sub_question in pp.sub_questions:
            ret = self.retriever(
                question=sub_question,
                document=document,
                history=history.as_text(),
            )
            retrieved.append(QAPair(question=sub_question, answer=str(ret.answer)))
        calc = self.calculator(question=question, retrieved=retrieved, program=pp.program)
        return AgentResponse(
            answer=str(calc.answer),
            turn_type=turn_type,
            conv_type=conv_type,
            program=str(pp.program),
            sub_questions=retrieved,
        )


def run_record(orchestrator: Any, record: ConvFinQARecord) -> list[dict[str, Any]]:
    history = ConversationHistory()
    rows: list[dict[str, Any]] = []
    document = serialize_document(record)
    for question, gold in zip(
        record.dialogue.conv_questions,
        record.dialogue.executed_answers,
        strict=False,
    ):
        pred = orchestrator(question=question, document=document, history=history)
        rows.append({"question": question, "pred_answer": pred.answer, "gold_answer": gold})
        history.append(question, str(gold), record.id)
    return rows
