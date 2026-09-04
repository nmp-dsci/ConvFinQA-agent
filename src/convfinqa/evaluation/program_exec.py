"""Execute a ConvFinQA DSL program.

The repo could parse and normalise programs but never run one: the pipeline
computes through tool calls in `pipeline/tools.py`, and `metrics.py` compares
programs textually. Attribution needs the third thing — *does this plan, given
these values, produce the gold answer?* — because the planned program is
**symbolic**. Preprocess emits `divide(A, B)`, where the placeholders bind to
its own sub-questions and only acquire values once the retriever answers. There
is no way to judge such a plan by its shape: `divide(A, B)` is right or wrong
depending entirely on what A and B turned out to be.

So `bind_and_execute` is what separates "preprocess planned something that
cannot reach gold" from "the plan was fine and the calculator fumbled it" — the
distinction the old skeleton comparison could not draw, and got backwards often
enough to misdirect two campaign experiments.
"""

from __future__ import annotations

import re
from typing import Any

from convfinqa.evaluation.metrics import parse_program

#: `parse_program` lowercases arguments, so placeholders arrive as `a`, `b`, ….
PLACEHOLDERS = "abcdefgh"

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: (a / b) if b else None,
    "exp": lambda a, b: a**b,
    "greater": lambda a, b: a > b,
}


def to_number(text: Any) -> float | None:
    """The first number in `text`, with a trailing `%` read as a proportion."""
    raw = str(text)
    match = _NUM_RE.search(raw.replace("%", ""))
    if not match:
        return None
    try:
        value = float(match.group().replace(",", ""))
    except ValueError:
        return None
    return value / 100.0 if "%" in raw else value


def execute(program: Any, binding: dict[str, float] | None = None) -> Any:
    """Run `program`, resolving `#N` back-references and bound placeholders.

    Returns the last step's value, or ``None`` if the program cannot be run —
    an unknown op, an argument that is neither a number nor bound, or a
    reference to a step that does not exist. ``None`` means *undecidable*, and
    callers must not read it as "the plan is wrong": those are different claims
    and only one of them should cost an agent a fault.
    """
    binding = binding or {}
    ops = parse_program(program)
    if not ops:
        return None
    steps: list[Any] = []
    for name, args in ops:
        fn = _OPS.get(name)
        if fn is None:
            return None
        values: list[float] = []
        for arg in args:
            token = str(arg).strip()
            if token.startswith("#"):
                try:
                    values.append(steps[int(token[1:])])
                except (ValueError, IndexError):
                    return None
            elif token in binding:
                values.append(binding[token])
            else:
                number = to_number(token)
                if number is None:
                    return None
                values.append(number)
        if len(values) != 2:
            return None
        try:
            result = fn(values[0], values[1])
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            return None
        if result is None:
            return None
        steps.append(result)
    return steps[-1] if steps else None


def bindings_from(answers: list[str]) -> dict[str, float]:
    """Bind placeholders to the retriever's answers **in sub-question order**.

    That ordering is the contract between preprocess and the retriever: the
    n-th placeholder is the n-th sub-question. An answer with no number in it
    binds nothing, which is what makes a declined sub-question show up as an
    unbindable plan rather than as a silently wrong one.
    """
    out: dict[str, float] = {}
    for i, answer in enumerate(answers):
        if i >= len(PLACEHOLDERS):
            break
        number = to_number(answer)
        if number is not None:
            out[PLACEHOLDERS[i]] = number
    return out


def result_matches(result: Any, gold_answer: Any) -> bool:
    """Compare an executed result to a gold answer, tolerating scale."""
    from convfinqa.evaluation.metrics import numeric_match

    if result is None:
        return False
    if isinstance(result, bool):
        return str(gold_answer).strip().lower() == ("yes" if result else "no")
    # Scale slack, because the dataset mixes conventions freely: a ratio against
    # a percentage gold, thousands against millions. `numeric_match` owns the
    # rounding tolerance and the `%` handling, so defer to it at each scale
    # rather than re-implementing either here.
    return any(
        numeric_match(float(result) * scale, gold_answer)
        for scale in (1.0, 100.0, 0.01, 1000.0, 0.001)
    )


def bind_and_execute(program: Any, answers: list[str], gold_answer: Any) -> bool | None:
    """Does `program`, bound to `answers`, reach `gold_answer`?

    ``None`` when the plan could not be bound and run at all — see `execute`.
    """
    result = execute(program, bindings_from(answers))
    if result is None:
        return None
    return result_matches(result, gold_answer)
