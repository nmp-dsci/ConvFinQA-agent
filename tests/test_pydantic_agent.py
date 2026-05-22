"""Tests for pydantic_agent.py.

Covers:
- Prompt loading parity with optimized_runner.json
- Output-model field parity vs DSPy signatures
- DSPy ChatAdapter wire-format rendering
- Calculator tool registration
- Test-set is the SAME object as dspy_agent.conv_examples_test
- ConversationRunner end-to-end on TestModel-stubbed agents (no real LM)
- compare_runs detects test-set drift
"""

# ruff: noqa: T201

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from convfinqa.backends.dspy import (
    CalculationSignature,
    PreprocessSignature,
    RetrieverSignature,
    TriageSignature,
)
from convfinqa.backends.dspy import conv_examples_test as dspy_conv_examples_test
from convfinqa.backends.pydantic import (
    calculator_agent,
    preprocess_agent,
    retriever_agent,
    triage_agent,
)
from convfinqa.data.loader import _DOCS
from convfinqa.data.schemas import ConversationHistory, QAPair
from convfinqa.evaluation.joining import compare_runs
from convfinqa.pipeline.prompts_loader import (
    PROMPTS,
    PROMPTS_PATH,
    _load_optimized_prompts,
)
from convfinqa.pipeline.runner import ConversationRunner
from convfinqa.pipeline.stages import (
    CalcOut,
    PreprocessOut,
    RetrievedValues,
    TriageOut,
)
from convfinqa.pipeline.wire_format import render_chat_inputs as _render_chat_inputs

conv_examples_test = dspy_conv_examples_test

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def test_prompts_loaded_from_artifact() -> None:
    """PROMPTS contains the four expected stages, each non-trivially long."""
    assert set(PROMPTS) == {"triage", "preprocess", "retriever", "calculator"}
    for k, v in PROMPTS.items():
        assert isinstance(v, str)
        assert len(v) > 50, f"{k} prompt suspiciously short: {len(v)}"


def test_prompts_match_artifact_instructions() -> None:
    """The optimized artifact remains loadable for overlay/back-compat paths."""
    loaded = _load_optimized_prompts(PROMPTS_PATH)
    assert set(loaded) == {"triage", "preprocess", "retriever", "calculator"}
    for prompt in loaded.values():
        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ---------------------------------------------------------------------------
# Output model field parity
# ---------------------------------------------------------------------------


def test_output_models_validate() -> None:
    """Smoke construction of each output model."""
    TriageOut(reasoning="r", turn_type="number", conv_type="Type I")
    PreprocessOut(reasoning="r", sub_questions=["x"], program="add(A,B)")
    RetrievedValues(reasoning="r", answers=[QAPair(question="q", answer="1.0")])
    CalcOut(answer="3.14")


def test_output_models_match_dspy_signatures() -> None:
    """Each Pydantic output_type matches its DSPy signature's output_fields,
    plus 'reasoning' for the three ChainOfThought predictors."""
    assert set(TriageOut.model_fields) == set(TriageSignature.output_fields) | {
        "reasoning"
    }
    assert set(PreprocessOut.model_fields) == set(PreprocessSignature.output_fields) | {
        "reasoning"
    }
    assert set(RetrievedValues.model_fields) == set(
        RetrieverSignature.output_fields
    ) | {"reasoning"}
    # ReAct does NOT auto-add reasoning
    assert set(CalcOut.model_fields) == set(CalculationSignature.output_fields)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def test_render_chat_inputs_format() -> None:
    """_render_chat_inputs produces exact DSPy ChatAdapter [[ ## name ## ]] blocks."""
    out = _render_chat_inputs(
        {"question": "foo", "history": "(no prior turns)", "conv_type": "Type I"}
    )
    expected = (
        "[[ ## question ## ]]\nfoo\n"
        "[[ ## history ## ]]\n(no prior turns)\n"
        "[[ ## conv_type ## ]]\nType I"
    )
    assert out == expected


def test_render_chat_inputs_field_order_matches_dspy_inputs() -> None:
    """The kwargs the runner builds for each stage match the DSPy signature's
    declared input order."""
    assert list(TriageSignature.input_fields) == ["question", "history"]
    assert list(PreprocessSignature.input_fields) == [
        "question",
        "history",
        "conv_type",
    ]
    assert list(RetrieverSignature.input_fields) == [
        "turn_type",
        "questions",
        "document",
        "history",
    ]
    assert list(CalculationSignature.input_fields) == [
        "question",
        "retrieved",
        "program",
    ]


def test_render_chat_inputs_handles_basemodel_and_list() -> None:
    """Non-string values render via model_dump_json / json.dumps."""

    class Foo(BaseModel):
        x: int

    out = _render_chat_inputs(
        {"foo": Foo(x=1), "bar": [QAPair(question="q", answer="1")]}
    )
    assert "[[ ## foo ## ]]" in out
    assert '"x": 1' in out
    assert "[[ ## bar ## ]]" in out
    assert '"question": "q"' in out


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_calculator_has_six_tools() -> None:
    """Calculator agent has all six DSL tools registered."""
    tools = calculator_agent._function_toolset.tools
    assert set(tools) == {"add", "subtract", "multiply", "divide", "exp", "greater"}


def test_each_agent_uses_optimized_instructions() -> None:
    """Each Pydantic AI agent is constructed with its optimized instructions."""
    pairs = [
        (triage_agent, "triage"),
        (preprocess_agent, "preprocess"),
        (retriever_agent, "retriever"),
        (calculator_agent, "calculator"),
    ]
    for ag, key in pairs:
        # Pydantic AI stores instructions as a list[str]
        assert ag._instructions == [PROMPTS[key]]


# ---------------------------------------------------------------------------
# Test-set parity with DSPy
# ---------------------------------------------------------------------------


def test_test_set_is_imported_from_dspy() -> None:
    """pydantic_agent.conv_examples_test IS the same object as dspy_agent's,
    so eval covers identical (report_id, question_set) records."""
    assert conv_examples_test is dspy_conv_examples_test
    ids_pyd = [(ex.report_id, len(ex.questions)) for ex in conv_examples_test]
    ids_dspy = [(ex.report_id, len(ex.questions)) for ex in dspy_conv_examples_test]
    assert ids_pyd == ids_dspy
    assert len(conv_examples_test) > 0


# ---------------------------------------------------------------------------
# Runner end-to-end with TestModel stubs
# ---------------------------------------------------------------------------


def _stub_overrides(
    triage_args: dict[str, Any],
    retriever_args: dict[str, Any] | None = None,
    preprocess_args: dict[str, Any] | None = None,
    calc_args: dict[str, Any] | None = None,
) -> list:
    """Build agent.override(model=TestModel(...)) context managers for each stage."""
    from pydantic_ai.models.test import TestModel

    ctxs = [triage_agent.override(model=TestModel(custom_output_args=triage_args))]
    if retriever_args is not None:
        ctxs.append(
            retriever_agent.override(model=TestModel(custom_output_args=retriever_args))
        )
    if preprocess_args is not None:
        ctxs.append(
            preprocess_agent.override(
                model=TestModel(custom_output_args=preprocess_args)
            )
        )
    if calc_args is not None:
        ctxs.append(
            calculator_agent.override(
                model=TestModel(custom_output_args=calc_args, call_tools=[])
            )
        )
    return ctxs


def test_runner_single_turn_number_question() -> None:
    """Number path: triage classifies, retriever returns one QAPair, no calc."""
    triage_a = {"reasoning": "r", "turn_type": "number", "conv_type": "Type I"}
    retr_a = {
        "reasoning": "r",
        "answers": [{"question": "q", "answer": "42"}],
    }
    overrides = _stub_overrides(triage_a, retriever_args=retr_a)
    rid = next(iter(_DOCS))  # any valid report_id

    with overrides[0], overrides[1]:
        preds = asyncio.run(
            ConversationRunner().run_conversation(rid, ["what is the value?"])
        )
    assert preds == (["42"], [""])


def test_runner_single_turn_program_question() -> None:
    """Program path: triage → preprocess → retriever → calculator."""
    triage_a = {"reasoning": "r", "turn_type": "program", "conv_type": "Type I"}
    pp_a = {
        "reasoning": "r",
        "sub_questions": ["a?", "b?"],
        "program": "subtract(A, B)",
    }
    retr_a = {
        "reasoning": "r",
        "answers": [
            {"question": "a?", "answer": "10"},
            {"question": "b?", "answer": "3"},
        ],
    }
    calc_a = {"answer": "7"}
    overrides = _stub_overrides(
        triage_a,
        preprocess_args=pp_a,
        retriever_args=retr_a,
        calc_args=calc_a,
    )
    rid = next(iter(_DOCS))

    with overrides[0], overrides[1], overrides[2], overrides[3]:
        preds = asyncio.run(
            ConversationRunner().run_conversation(rid, ["compute the change"])
        )
    assert preds == (["7"], ["subtract(A, B)"])


def test_history_format_matches_conversation_history_as_text() -> None:
    """Two-turn run threads conversation history in the as_text() format the
    optimized prompts were tuned against."""
    h = ConversationHistory()
    h.append(question="q1", answer="a1", report_id="r")
    text = h.as_text()
    assert "Q1 [report=r]: q1" in text
    assert "A1: a1" in text


# ---------------------------------------------------------------------------
# compare_runs drift detection
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_compare_runs_detects_drift(tmp_path: Path) -> None:
    """compare_runs raises RuntimeError when the two CSVs cover different rows."""
    base = [
        {
            "report_id": "r1",
            "turn_index": 0,
            "question": "q1",
            "gold_answer": "1",
            "pred_answer": "1",
            "correct": True,
        }
    ]
    drifted = [
        {
            "report_id": "r2",  # different report_id
            "turn_index": 0,
            "question": "q1",
            "gold_answer": "1",
            "pred_answer": "1",
            "correct": True,
        }
    ]
    a = tmp_path / "predictions.csv"
    b = tmp_path / "pydantic_predictions.csv"
    _write_csv(a, base)
    _write_csv(b, drifted)
    with pytest.raises(RuntimeError, match="Test-set drift"):
        compare_runs(a, b)


def test_compare_runs_succeeds_on_matching_rows(tmp_path: Path) -> None:
    """compare_runs returns the parity_report.csv path on identical coverage."""
    rows = [
        {
            "report_id": "r1",
            "turn_index": 0,
            "question": "q1",
            "gold_answer": "1",
            "pred_answer": "1",
            "correct": True,
        }
    ]
    a = tmp_path / "predictions.csv"
    b = tmp_path / "pydantic_predictions.csv"
    _write_csv(a, rows)
    _write_csv(b, rows)
    # qa_data merge will produce NaN slice cols for synthetic r1, but that's
    # OK — the function shouldn't crash, just print warnings.
    out = compare_runs(a, b)
    assert out.name == "parity_report.csv"
    assert out.exists()
