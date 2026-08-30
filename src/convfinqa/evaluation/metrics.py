"""Matching rules for ConvFinQA predictions: the answer, and the reasoning.

Two oracles live here, and the difference between them is the point.

`numeric_match(pred, gold)` is **execution accuracy** — did the final number come
out right. It is the headline metric and it is also the one that can be passed by
accident: a two-step program with two compensating errors, or a value guessed off
the right row for the wrong reason, both score a point.

`program_match(pred, gold)` is **program accuracy** — did the system reach that
number the way the annotator did. The ConvFinQA paper reports both for exactly
this reason, and the gap between them is the honest measure of how much of the
execution score is reasoning and how much is luck. Normalisation follows the
paper: commutative arguments are order-insensitive, `const_*` tokens are resolved
to their values, and nested calls are flattened to the `#n` reference form so two
spellings of one program compare equal.

Keep all matching logic here so the three runners (api_eval, pydantic_agent,
dspy_agent) cannot diverge.
"""

from __future__ import annotations

import math
import re
from typing import Any


def _to_float(s: str) -> tuple[float, bool]:
    """Parse a numeric string, detecting a trailing % suffix.

    Returns (numeric_value, is_percentage_notation).
    """
    s = s.strip()
    is_pct = s.endswith("%")
    return float(s.rstrip("%").strip()), is_pct


def numeric_match(pred: Any, gold: Any) -> bool:
    """Return True if pred and gold represent the same numeric answer.

    Rules:

    1. Both values round to the same integer — handles "60%" vs "59.7%" and
       "119%" vs "118.9%".
    2. Explicit percentage vs decimal mismatch: one side carries a "%" suffix
       and the other does not. Convert the percentage to a decimal (÷ 100) and
       compare at 3 decimal places — e.g. "90.9%" matches "0.9091".
    3. Implicit percentage vs decimal: neither side has a "%" suffix but one
       value is ≤ 1 (decimal proportion) and the other is ~100x larger.
       Dividing the larger by 100 and comparing at 3 dp catches models that
       output "90.9" when the gold answer is "0.9091".

    Non-numeric inputs fall back to case-insensitive string comparison.

    A NaN or infinite value on either side is never a match. `float("nan")`
    parses cleanly and then explodes in `round()`, so it has to be rejected here
    rather than caught by the parse guard — a missing prediction reaches this
    function as the literal string "nan" whenever a conversation errored out and
    its row was written with an empty answer.
    """
    try:
        pv, p_is_pct = _to_float(str(pred))
        gv, g_is_pct = _to_float(str(gold))
    except (ValueError, TypeError):
        return str(pred).strip().lower() == str(gold).strip().lower()

    if not (math.isfinite(pv) and math.isfinite(gv)):
        return False

    # Rule 1: integer-rounded equality
    if round(pv) == round(gv):
        return True

    # Rule 2: explicit % notation on exactly one side
    if p_is_pct and not g_is_pct and round(pv / 100, 3) == round(gv, 3):
        return True
    if g_is_pct and not p_is_pct and round(pv, 3) == round(gv / 100, 3):
        return True

    # Rule 3: implicit pct vs decimal — smaller value ≤ 1, larger ÷ 100 matches at 3 dp
    if not p_is_pct and not g_is_pct and pv != 0 and gv != 0:
        larger, smaller = (pv, gv) if abs(pv) > abs(gv) else (gv, pv)
        if abs(smaller) <= 1 and round(larger / 100, 3) == round(smaller, 3):
            return True

    return False


# ---------------------------------------------------------------------------
# Program accuracy
# ---------------------------------------------------------------------------

#: The DSL's operations. `greater` returns a bool; the rest return numbers.
OPERATIONS: frozenset[str] = frozenset(
    {"add", "subtract", "multiply", "divide", "exp", "greater"}
)

#: Operations whose arguments may be reordered without changing the program.
#: `subtract` and `divide` are excluded for the obvious reason, and `exp` and
#: `greater` because base/exponent and left/right are not interchangeable either.
COMMUTATIVE: frozenset[str] = frozenset({"add", "multiply"})

_CONST_RE = re.compile(r"^const_(m?)([0-9]+(?:\.[0-9]+)?)$")
_NUM_CLEAN_RE = re.compile(r"[,$%\s]")


def _canon_arg(token: str) -> str:
    """Canonicalise one program argument.

    Three normalisations, in the order the paper implies:

    1. `const_100` / `const_m1` become plain values, so a gold program written
       with the annotation vocabulary compares against a prediction written in
       arithmetic.
    2. `#0` references pass through untouched — they are structure, not value.
    3. Everything else that parses as a number is reduced to one spelling, so
       `1,234.0`, `$1234` and `1234` are one argument rather than three.
    """
    token = token.strip()
    if not token:
        return ""
    if token.startswith("#"):
        return token
    const = _CONST_RE.match(token.lower())
    if const:
        sign, value = const.groups()
        token = ("-" if sign else "") + value
    cleaned = _NUM_CLEAN_RE.sub("", token)
    try:
        value_f = float(cleaned)
    except ValueError:
        return token.lower()
    if not math.isfinite(value_f):
        return token.lower()
    # `243` and `243.0` are the same argument; `%g` collapses them without
    # rounding anything a financial figure would care about.
    return f"{value_f:.10g}"


def _split_args(body: str) -> list[str]:
    """Split a call's argument list on top-level commas only."""
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            args.append("".join(current))
            current = []
            continue
        current.append(char)
    if current or args:
        args.append("".join(current))
    return [a.strip() for a in args]


def parse_program(program: Any) -> list[tuple[str, list[str]]] | None:
    """Parse a DSL program into a flat op list, or None if it is not a program.

    Accepts both notations the corpus contains: the annotator's flat sequence
    (`subtract(243, 111), divide(#0, 111)`) and the nested spelling a model
    tends to emit (`divide(subtract(243, 111), 111)`). Nested calls are hoisted
    into the flat form, so the two parse identically — which is the whole reason
    program accuracy is worth computing rather than string-comparing.

    Returns None — not an empty list — when the text holds no operation at all.
    A number-selection turn's "program" is its answer, and scoring that as a
    failed program would quietly depress the metric with turns the paper does
    not count.
    """
    if program is None:
        return None
    text = str(program).strip()
    if not text or text.lower() == "nan":
        return None

    ops: list[tuple[str, list[str]]] = []

    def emit(call: str) -> str:
        """Parse one call, appending it (and its nested calls) to `ops`."""
        call = call.strip()
        open_at = call.find("(")
        if open_at < 0 or not call.endswith(")"):
            return call
        name = call[:open_at].strip().lower()
        if name not in OPERATIONS:
            return call
        args = [
            emit(arg) if "(" in arg else arg
            for arg in _split_args(call[open_at + 1 : -1])
        ]
        ops.append((name, [_canon_arg(a) for a in args]))
        return f"#{len(ops) - 1}"

    for call in _split_args(text):
        emit(call)

    return ops or None


def normalize_program(program: Any) -> str:
    """Canonical string form of a program, or `""` when there is none.

    Two programs are equivalent exactly when their normal forms are equal, which
    makes this the thing to log or diff when a program-accuracy result surprises
    you.
    """
    ops = parse_program(program)
    if not ops:
        return ""
    parts = []
    for name, args in ops:
        ordered = sorted(args) if name in COMMUTATIVE else args
        parts.append(f"{name}({','.join(ordered)})")
    return ", ".join(parts)


def program_match(pred: Any, gold: Any) -> bool:
    """True when `pred` is the same program as `gold` under the paper's rules.

    An unparseable or absent prediction is a miss, never a match: the metric
    exists to catch a right answer reached the wrong way, so it must not credit
    an answer reached by no visible way at all.
    """
    gold_norm = normalize_program(gold)
    if not gold_norm:
        return False
    return normalize_program(pred) == gold_norm


def has_program(gold: Any) -> bool:
    """True when a gold entry is a real program rather than a selected number."""
    return bool(parse_program(gold))


def program_from_trajectory(trajectory: Any) -> str:
    """Rebuild the executed program from a calculator's tool-call trace.

    The pipeline's `pred_program` is written by the *preprocess* stage over
    sub-question placeholders (`multiply(divide(C, B), 100)`), so it carries the
    shape of the reasoning but not its values, and cannot be compared to a gold
    program directly. The calculator's trajectory is the same program after the
    retriever filled it in — real operations over real numbers — which is what
    the paper's program accuracy actually scores.

    Returns `""` when the trace holds no arithmetic (a number turn, or an
    errored one).
    """
    if not isinstance(trajectory, list):
        return ""
    ops: list[str] = []
    results: dict[str, str] = {}
    for index, step in enumerate(trajectory):
        if not isinstance(step, dict) or step.get("event") != "tool_call":
            continue
        name = str(step.get("tool", "")).lower()
        if name not in OPERATIONS:
            continue
        args = step.get("args")
        if not isinstance(args, dict):
            continue
        rendered: list[str] = []
        for value in args.values():
            canon = _canon_arg(str(value))
            # A tool call whose argument is a previous call's return value is a
            # `#n` reference in DSL terms; recovering that is what turns a flat
            # tool log back into a program.
            rendered.append(results.get(canon, canon))
        ops.append(f"{name}({','.join(rendered)})")
        key = _result_key(trajectory, index)
        if key:
            results[key] = f"#{len(ops) - 1}"
    return ", ".join(ops)


def _result_key(trajectory: list[Any], call_index: int) -> str:
    """Canonical form of the value the tool_call at `call_index` returned.

    Matched by position rather than by value: two calls in one trajectory can
    return the same number, and linking them by value would make the second
    reference point at the first call.
    """
    call = trajectory[call_index]
    for step in trajectory[call_index + 1 :]:
        if not isinstance(step, dict) or step.get("event") != "tool_return":
            continue
        if str(step.get("tool", "")).lower() == str(call.get("tool", "")).lower():
            return _canon_arg(str(step.get("result", "")))
        break
    return ""
