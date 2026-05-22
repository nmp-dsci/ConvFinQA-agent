"""Numeric matching rules for ConvFinQA predictions.

`numeric_match(pred, gold)` returns True under any of the tolerance rules
documented below. The function is the canonical correctness oracle used by
every evaluation script in the project — keep all matching logic here so the
three runners (api_eval, pydantic_agent, dspy_agent) cannot diverge.
"""

from __future__ import annotations

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
    """
    try:
        pv, p_is_pct = _to_float(str(pred))
        gv, g_is_pct = _to_float(str(gold))
    except (ValueError, TypeError):
        return str(pred).strip().lower() == str(gold).strip().lower()

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
