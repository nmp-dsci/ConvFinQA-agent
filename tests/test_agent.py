"""Tests for agent.py — covers pure functions and orchestrator branching.

The orchestrator tests stub each inner agent with a small dspy.Module subclass
that returns a pre-built dspy.Prediction. This avoids any LM calls and keeps
the tests deterministic and fast.
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest

from tests._fixtures import legacy_agent as agent
from tests._fixtures.legacy_agent import (
    CALCULATOR_TOOLS,
    AgentResponse,
    ConversationHistory,
    ConvFinQAOrchestrator,
    QAPair,
    add,
    divide,
    exp,
    greater,
    multiply,
    numeric_match,
    sample_records,
    serialize_document,
    subtract,
)

# ---------------------------------------------------------------------------
# Calculator tools
# ---------------------------------------------------------------------------


def test_add() -> None:
    """add returns the arithmetic sum."""
    assert add(1.0, 2.0) == 3.0
    assert add(-1.0, 1.0) == 0.0


def test_subtract() -> None:
    """subtract returns the arithmetic difference."""
    assert subtract(5.0, 3.0) == 2.0


def test_multiply() -> None:
    """multiply returns the arithmetic product."""
    assert multiply(2.0, 3.5) == 7.0


def test_divide() -> None:
    """divide returns the quotient."""
    assert divide(10.0, 4.0) == 2.5


def test_divide_by_zero_raises() -> None:
    """divide raises ZeroDivisionError when b is zero."""
    with pytest.raises(ZeroDivisionError):
        divide(1.0, 0.0)


def test_exp() -> None:
    """exp returns a ** b."""
    assert exp(2.0, 10.0) == 1024.0


def test_greater_returns_bool() -> None:
    """greater returns a strict boolean."""
    assert greater(2.0, 1.0) is True
    assert greater(1.0, 2.0) is False
    assert greater(1.0, 1.0) is False


def test_calculator_tools_list_contents() -> None:
    """CALCULATOR_TOOLS contains the six expected callables."""
    assert CALCULATOR_TOOLS == [add, subtract, multiply, divide, exp, greater]


# ---------------------------------------------------------------------------
# ConversationHistory
# ---------------------------------------------------------------------------


def test_conversation_history_empty_text() -> None:
    """as_text returns the empty marker when there are no turns."""
    assert ConversationHistory().as_text() == "(no prior turns)"


def test_conversation_history_append_and_text() -> None:
    """as_text formats each turn as 'Q{i}: ...\\nA{i}: ...' starting at 1."""
    h = ConversationHistory()
    h.append("what was net income?", "206588")
    h.append("and in 2008?", "181001")
    text = h.as_text()
    assert "Q1: what was net income?" in text
    assert "A1: 206588" in text
    assert "Q2: and in 2008?" in text
    assert "A2: 181001" in text


# ---------------------------------------------------------------------------
# numeric_match
# ---------------------------------------------------------------------------


def test_numeric_match_within_tolerance() -> None:
    """Numeric values within 1e-3 are treated as equal."""
    assert numeric_match("0.14136", 0.1414) is True


def test_numeric_match_outside_tolerance() -> None:
    """Numeric values outside 1e-3 are not equal."""
    assert numeric_match("0.140", 0.150) is False


def test_numeric_match_exact_string_fallback() -> None:
    """Non-numeric predictions fall back to string equality."""
    assert numeric_match("yes", "yes") is True
    assert numeric_match("yes", "no") is False


def test_numeric_match_handles_negative_and_int() -> None:
    """Negative numbers and ints are compared numerically."""
    assert numeric_match("-25587", -25587.0) is True
    assert numeric_match("206588", 206588) is True


# ---------------------------------------------------------------------------
# sample_records — reproducibility
# ---------------------------------------------------------------------------


def test_sample_records_is_reproducible() -> None:
    """Same seed -> same record ids in the same order, run twice."""
    a = [r.id for r in sample_records(n=2, seed=42)]
    b = [r.id for r in sample_records(n=2, seed=42)]
    assert a == b
    assert len(a) == 2


def test_sample_records_different_seed_differs() -> None:
    """Different seeds produce different selections (sanity check)."""
    a = [r.id for r in sample_records(n=2, seed=42)]
    b = [r.id for r in sample_records(n=2, seed=123)]
    assert a != b


def test_sample_records_count() -> None:
    """sample_records honours the n argument."""
    assert len(sample_records(n=3, seed=42)) == 3


# ---------------------------------------------------------------------------
# serialize_document
# ---------------------------------------------------------------------------


def test_serialize_document_contains_sections() -> None:
    """Serialised document contains each of the three section headers."""
    rec = sample_records(n=1, seed=42)[0]
    text = serialize_document(rec)
    assert "PRE-TEXT:" in text
    assert "TABLE:" in text
    assert "POST-TEXT:" in text


# ---------------------------------------------------------------------------
# Orchestrator — stub-module pattern
# ---------------------------------------------------------------------------


class _StubModule(dspy.Module):
    """dspy.Module subclass that always returns a pre-built Prediction."""

    def __init__(self, pred: dspy.Prediction) -> None:
        """Store the canned prediction."""
        super().__init__()
        self.pred = pred

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Return the canned prediction, ignoring inputs."""
        return self.pred


class _SequenceModule(dspy.Module):
    """dspy.Module subclass that returns a different Prediction per call."""

    def __init__(self, preds: list[dspy.Prediction]) -> None:
        """Store the queue of canned predictions."""
        super().__init__()
        self.preds = list(preds)
        self.i = 0

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Return predictions in order; raises IndexError if exhausted."""
        p = self.preds[self.i]
        self.i += 1
        return p


def _stub_orchestrator(
    triage: dspy.Prediction,
    preprocess: dspy.Prediction | None = None,
    retriever: dspy.Prediction | list[dspy.Prediction] | None = None,
    calculator: dspy.Prediction | None = None,
) -> ConvFinQAOrchestrator:
    """Build a ConvFinQAOrchestrator with each sub-agent stubbed out."""
    orch = ConvFinQAOrchestrator()
    orch.triage = _StubModule(triage)
    if preprocess is not None:
        orch.preprocess = _StubModule(preprocess)
    if retriever is not None:
        if isinstance(retriever, list):
            orch.retriever = _SequenceModule(retriever)
        else:
            orch.retriever = _StubModule(retriever)
    if calculator is not None:
        orch.calculator = _StubModule(calculator)
    return orch


def test_orchestrator_number_path() -> None:
    """When triage returns turn_type=number, only retriever runs."""
    orch = _stub_orchestrator(
        triage=dspy.Prediction(turn_type="number", conv_type="Type I"),
        retriever=dspy.Prediction(answer="206588"),
    )
    resp = orch(
        question="what was net income in 2009?",
        document="(doc)",
        history=ConversationHistory(),
    )
    assert isinstance(resp, AgentResponse)
    assert resp.answer == "206588"
    assert resp.turn_type == "number"
    assert resp.conv_type == "Type I"
    assert resp.program is None
    assert resp.sub_questions == [
        QAPair(question="what was net income in 2009?", answer="206588"),
    ]


def test_orchestrator_program_path() -> None:
    """When triage returns turn_type=program, preprocess + retriever (xN) + calculator all run."""
    orch = _stub_orchestrator(
        triage=dspy.Prediction(turn_type="program", conv_type="Type I"),
        preprocess=dspy.Prediction(
            sub_questions=["net income 2009?", "net income 2008?"],
            program="subtract(A, B)",
        ),
        retriever=[
            dspy.Prediction(answer="206588"),
            dspy.Prediction(answer="181001"),
        ],
        calculator=dspy.Prediction(answer="25587"),
    )
    resp = orch(
        question="what was the change in net income?",
        document="(doc)",
        history=ConversationHistory(),
    )
    assert isinstance(resp, AgentResponse)
    assert resp.answer == "25587"
    assert resp.turn_type == "program"
    assert resp.conv_type == "Type I"
    assert resp.program == "subtract(A, B)"
    assert len(resp.sub_questions) == 2
    assert resp.sub_questions[0] == QAPair(question="net income 2009?", answer="206588")
    assert resp.sub_questions[1] == QAPair(question="net income 2008?", answer="181001")


def test_orchestrator_passes_history_to_retriever() -> None:
    """Conversation history reaches the retriever as formatted text."""
    captured: dict[str, str] = {}

    class _Capture(dspy.Module):
        def __init__(self, pred: dspy.Prediction) -> None:
            super().__init__()
            self.pred = pred

        def forward(self, **kwargs: Any) -> dspy.Prediction:
            captured.update(kwargs)
            return self.pred

    orch = ConvFinQAOrchestrator()
    orch.triage = _StubModule(dspy.Prediction(turn_type="number", conv_type="Type I"))
    orch.retriever = _Capture(dspy.Prediction(answer="x"))

    history = ConversationHistory()
    history.append("what was net income?", "206588")
    orch(question="and in 2008?", document="(doc)", history=history)

    assert "Q1: what was net income?" in captured["history"]
    assert "A1: 206588" in captured["history"]


# ---------------------------------------------------------------------------
# run_record / evaluate plumbing
# ---------------------------------------------------------------------------


def test_run_record_uses_teacher_forced_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teacher-forced history: gold answers are appended between turns, not predictions."""
    rec = sample_records(n=1, seed=42)[0]

    seen_histories: list[str] = []

    def _fake_orch(
        question: str,
        document: str,
        history: ConversationHistory,
    ) -> AgentResponse:
        seen_histories.append(history.as_text())
        return AgentResponse(
            answer="WRONG",
            turn_type="number",
            conv_type="Type I",
        )

    rows = agent.run_record(_fake_orch, rec)  # type: ignore[arg-type]

    assert len(rows) == len(rec.dialogue.conv_questions)
    if len(rec.dialogue.conv_questions) >= 2:
        # The history seen on turn 2 must include the GOLD answer from turn 1
        # (not "WRONG", which is what the fake orchestrator returned).
        gold_t1 = str(rec.dialogue.executed_answers[0])
        assert gold_t1 in seen_histories[1]
        assert "WRONG" not in seen_histories[1]
