"""The checked open harness (s29, the owner's D5): three turn checks for the student.

``--open-harness pydantic_ai_checked`` is `open_agent`'s harness with three
checks a reply must pass, each found in epoch 2's test failures:

- **H1** a calculator call with an argument beyond ``a`` and ``b`` gets a tool
  error naming the fix, not a ``TypeError`` that ends the turn (4 test turns
  crashed on ``add(a, b, c)``).
- **H2** a program turn's answer is one of the turn's calculator results, as
  the prompt rounds or scales it, and a program turn calls the calculator at
  all (C/2009 answered 0.178 where the tool said 1.784).
- **H3** the answer is what the reply's own program gives on its own retrieved
  values, so the calculator ran the operands in the program's order
  (UNP/2013: program ``subtract(B, A)`` = −6, tool call ``subtract(637, 631)``).

H2 and H3 send one retry per question at most; a reply that fails again is
accepted as it stands, so a check can cost a question one extra round, never
its answer. The plain harness stays as it was: the teacher and every pass
before s29 ran it, and `pydantic_ai_checked` is a different run name.
"""

from __future__ import annotations

import re
from typing import Any

CHECKED_HARNESS = "pydantic_ai_checked"
CALCULATOR_PREFIX = "mcp__cfq__"

_NUMBER = re.compile(r"^\s*\$?(-?[\d,]*\.?(\d*))\s*%?\s*$")


def answer_is_result(answer: Any, result: Any) -> bool:
    """Is `answer` the calculator's `result`, as the prompt says to write it?

    The prompt rounds (a percentage to one decimal, say) and may scale by 100,
    so the answer is the result rounded to the answer's own precision, at a
    scale of 1, 100 or 1/100. A comparison's ``True``/``False`` is yes/no.
    """
    a, r = str(answer).strip(), str(result).strip()
    if r in ("True", "False"):
        return a.lower() in ({"yes", "true"} if r == "True" else {"no", "false"})
    m = _NUMBER.match(a)
    try:
        value = float(m.group(1).replace(",", "")) if m else None
        exact = float(r)
    except ValueError:
        return False
    if value is None:
        return False
    decimals = len(m.group(2)) if m else 0
    return any(
        abs(round(exact * scale, decimals) - value) < 10 ** -(decimals + 6)
        or abs(exact * scale - value) <= 1e-9 * max(1.0, abs(value))
        for scale in (1.0, 100.0, 0.01)
    )


def extra_argument_error(name: str, extra: dict[str, Any]) -> str:
    """H1's tool error: what was wrong and how to chain instead."""
    names = ", ".join(sorted(extra))
    return (
        f"error: {name} takes exactly two numbers, a and b; it was also given "
        f"{names}. For three or more numbers, chain calls: {name}(a, b), then "
        f"{name}(that result, c)."
    )


def turn_results(messages: list[Any]) -> list[str]:
    """The calculator results of the question being answered, oldest first.

    The question starts at the last request holding a user prompt; an
    ``error: …`` return is not a result.
    """
    from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

    start = 0
    for i, message in enumerate(messages):
        if isinstance(message, ModelRequest) and any(
            isinstance(p, UserPromptPart) for p in message.parts
        ):
            start = i
    out: list[str] = []
    for message in messages[start:]:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if (
                isinstance(part, ToolReturnPart)
                and part.tool_name.startswith(CALCULATOR_PREFIX)
                and not str(part.content).startswith("error:")
            ):
                out.append(str(part.content))
    return out


def program_values(output: Any) -> list[Any]:
    """What the reply's program gives on its own retrieved values.

    Two readings of a ``12%`` operand, as a proportion and as written, since
    the calculator may have been given either; empty when the program cannot
    be bound and run at all.
    """
    from convfinqa.evaluation.program_exec import (
        PLACEHOLDERS,
        _raw_number,
        bindings_from,
        execute,
    )

    answers = [str(r.answer) for r in output.retrieved]
    raw = {
        PLACEHOLDERS[i]: n
        for i, a in enumerate(answers[: len(PLACEHOLDERS)])
        if (n := _raw_number(a)) is not None
    }
    values = (
        execute(output.program, bindings_from(answers)),
        execute(output.program, raw),
    )
    return [v for v in values if v is not None]


def failed_check(output: Any, messages: list[Any]) -> tuple[str, str] | None:
    """The first check `output` fails (H2, then H3) and its retry message, or None."""
    if str(output.turn_type) != "program":
        return None
    results = turn_results(messages)
    if not results:
        return "H2", (
            "This is a program turn, so its arithmetic goes through the calculator "
            "tools: call them, then answer with the last result."
        )
    if not any(answer_is_result(output.answer, r) for r in results):
        shown = ", ".join(results[-4:])
        return "H2", (
            f"Your answer {output.answer!r} is not any calculator result from this "
            f"question (results: {shown}). Answer with the calculator's result, "
            "rounded as the instructions say."
        )
    values = program_values(output)
    if values and not any(answer_is_result(output.answer, v) for v in values):
        operands = ", ".join(str(r.answer) for r in output.retrieved)
        return "H3", (
            f"Your program {output.program} on your retrieved values ({operands}) "
            f"gives {values[0]}, but your answer is {output.answer!r}. Call the "
            "calculator with the operands in the program's order and answer with "
            "its result."
        )
    return None
