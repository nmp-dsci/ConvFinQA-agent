"""Program accuracy: did it reach the number the way the annotator did.

Execution accuracy can be passed by accident — two compensating errors, or the
right value read off the right row for the wrong reason. Program accuracy is the
control on that, and it is only worth having if its normalisation is right, so
the normalisation is what these tests pin: what counts as the same program, what
does not, and which turns are scored at all.
"""

from __future__ import annotations

import pandas as pd

from convfinqa.evaluation.metrics import (
    has_program,
    normalize_program,
    parse_program,
    program_from_trajectory,
    program_match,
)
from convfinqa.tracking.comparator import program_accuracy

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_nested_and_flat_spellings_are_one_program() -> None:
    """The annotator writes `#0`; a model writes nesting. Same reasoning."""
    assert program_match(
        "divide(subtract(243.0, 111), 111.00)",
        "subtract(243, 111), divide(#0, 111)",
    )


def test_commutative_arguments_may_be_reordered() -> None:
    """`add(a, b)` and `add(b, a)` are the same step."""
    assert program_match("add(2, 1)", "add(1, 2)")
    assert program_match("multiply(3, 4)", "multiply(4, 3)")


def test_non_commutative_arguments_may_not() -> None:
    """`subtract` and `divide` are the ops the whole dataset turns on."""
    assert not program_match("subtract(1, 2)", "subtract(2, 1)")
    assert not program_match("divide(1, 2)", "divide(2, 1)")
    assert not program_match("exp(2, 3)", "exp(3, 2)")


def test_constants_are_normalised_to_their_values() -> None:
    """`const_100` is the annotation vocabulary for the number 100."""
    assert program_match("divide(#0, 100)", "divide(#0, const_100)")
    assert program_match("multiply(#0, -1)", "multiply(#0, const_m1)")


def test_number_formatting_is_not_a_difference() -> None:
    """`243`, `243.0` and `$243` are one argument, not three.

    A thousands separator is deliberately *not* stripped inside a program: the
    comma is the argument separator there, so `add(1,234, 1)` is three arguments
    and pretending otherwise would silently reinterpret a malformed program.
    Values arriving from a tool trace are cleaned; program text is not.
    """
    assert normalize_program("add(243, 1)") == normalize_program("add(243.0, 1.00)")
    assert normalize_program("add($243, 1)") == normalize_program("add(243, 1)")
    assert normalize_program("add(1,234, 1)") == "add(1,1,234)"


def test_a_selected_number_is_not_a_program() -> None:
    """A number turn's gold "program" is its answer; it must not be scored."""
    assert parse_program("243") is None
    assert parse_program("") is None
    assert parse_program(float("nan")) is None
    assert not has_program("243")
    assert has_program("subtract(243, 111)")


def test_a_missing_prediction_never_matches() -> None:
    """The metric exists to catch reasoning; absent reasoning is not a match."""
    assert not program_match("", "subtract(2, 1)")
    assert not program_match(float("nan"), "subtract(2, 1)")
    assert not program_match("subtract(2, 1)", "")


# ---------------------------------------------------------------------------
# Recovering the executed program from the calculator trace
# ---------------------------------------------------------------------------


def test_trajectory_becomes_a_referenced_program() -> None:
    """A flat tool log is a program once return values are linked back."""
    trajectory = [
        {"event": "tool_call", "tool": "divide", "args": {"a": 132, "b": 111}},
        {"event": "tool_return", "tool": "divide", "result": "1.1891891891891893"},
        {
            "event": "tool_call",
            "tool": "multiply",
            "args": {"a": 1.1891891891891893, "b": 100},
        },
        {"event": "tool_return", "tool": "multiply", "result": "118.9"},
        {"event": "tool_call", "tool": "final_result", "args": {"answer": "118.9%"}},
    ]
    assert program_from_trajectory(trajectory) == "divide(132,111), multiply(#0,100)"


def test_trajectory_without_arithmetic_yields_nothing() -> None:
    """A number turn has no program, and must not be given one."""
    assert program_from_trajectory([]) == ""
    assert program_from_trajectory(None) == ""
    assert (
        program_from_trajectory(
            [{"event": "tool_call", "tool": "final_result", "args": {"answer": "1"}}]
        )
        == ""
    )


# ---------------------------------------------------------------------------
# Frame-level roll-up
# ---------------------------------------------------------------------------


def _row(gold: str, pred: str, calculator_io: str = "") -> dict[str, object]:
    return {
        "report_id": "r1",
        "turn_index": 0,
        "correct": True,
        "gold_program": gold,
        "pred_program": pred,
        "calculator_io": calculator_io,
    }


def test_only_gold_programs_are_scored() -> None:
    """Number turns are excluded from the denominator, not counted as misses."""
    df = pd.DataFrame(
        [
            _row("subtract(2, 1)", "subtract(2, 1)"),
            _row("243", ""),  # a number turn
            _row("111", ""),  # a number turn
        ]
    )
    result = program_accuracy(df)
    assert result["n_program_turns"] == 1
    assert result["program_accuracy"] == 1.0


def test_placeholder_programs_fall_back_to_the_executed_trace() -> None:
    """`pred_program` is symbolic; the calculator trace is the real program."""
    calculator_io = (
        '{"trajectory": ['
        '{"event": "tool_call", "tool": "subtract", "args": {"a": 243, "b": 111}}'
        "]}"
    )
    df = pd.DataFrame([_row("subtract(243, 111)", "subtract(A, B)", calculator_io)])
    result = program_accuracy(df)
    assert result["n_program_turns"] == 1
    assert result["program_accuracy"] == 1.0


def test_empty_frame_scores_zero_rather_than_dividing_by_zero() -> None:
    """A frame with nothing to score is a valid input."""
    df = pd.DataFrame(columns=["gold_program", "pred_program", "calculator_io"])
    assert program_accuracy(df)["program_accuracy"] == 0.0


def test_committed_predictions_score_and_stay_below_execution_accuracy() -> None:
    """The real CSVs, scored offline. No API calls, and the gap is the finding.

    Program accuracy sitting well under execution accuracy is expected here and
    is not a bug: the pipeline answers a turn using prior *answers* from the
    conversation, while the gold program re-derives everything from raw values,
    so many correct answers are reached by a different — shorter — program. The
    assertion is on the relationship, which is what makes the pair informative,
    rather than on a number that would need re-running an eval to change.
    """
    from convfinqa.tracking.comparator import accuracy, load_predictions

    df = load_predictions("v2")
    result = program_accuracy(df)
    assert result["n_program_turns"] > 0
    assert 0.0 < result["program_accuracy"] < accuracy(df)
