"""Calculator tools shared by all backends."""

from __future__ import annotations

from typing import Any


def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return a / b. Raises ZeroDivisionError if b == 0."""
    return a / b


def exp(a: float, b: float) -> float:
    """Return a raised to the power b."""
    return float(a**b)


def greater(a: float, b: float) -> bool:
    """Return True iff a is strictly greater than b."""
    return a > b


CALCULATOR_TOOLS: list[Any] = [add, subtract, multiply, divide, exp, greater]
