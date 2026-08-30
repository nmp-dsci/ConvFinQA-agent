"""Stable error codes for every failure a turn can end in.

The trace store recorded `error` as free text, which is enough to *count* a
failure and not enough to explain one: "provider returned 503", "model call
exceeded 120s and was abandoned" and "no recording for that question" are three
unrelated conditions that a free-text column can only present as three strings.
An error-rate tile built on that can say 4% and nothing else.

So a small closed vocabulary sits beside the free text — never instead of it.
The code is what a dashboard groups by and what the frontend maps to its own
copy; the message is what a human reads when they open the row. Losing either
one makes the other worse.

The vocabulary is deliberately short. A code earns its place by implying a
different response: a rate limit means wait, a demo refusal means this
deployment will never do that, a missing recording means ask a different
question. Codes that would imply the same response stay merged under `unknown`,
where the free text does the explaining.
"""

from __future__ import annotations

import asyncio
from enum import Enum


class ErrorCode(str, Enum):
    """The closed set of codes a failed turn is classified into."""

    LLM_UNAVAILABLE = "llm_unavailable"
    NOT_AVAILABLE_DEMO = "not_available_demo"
    NO_RECORDING = "no_recording"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


#: Every value, for validation and for a dashboard that wants a stable x-axis.
ALL_CODES: tuple[str, ...] = tuple(code.value for code in ErrorCode)

_BY_VALUE = {code.value: code for code in ErrorCode}


def _status_of(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def classify(exc: BaseException) -> str:
    """Map an exception to one of `ErrorCode`'s values.

    Order matters. A timeout is checked before the exception's own `code`
    attribute because `llm.call_with_budget` reports an abandoned call as
    `LLMUnavailableError` — true, but `timeout` is the more useful of the two
    facts, and the free text keeps the other one.
    """
    if isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return ErrorCode.TIMEOUT.value

    status = _status_of(exc)
    if status == 429:
        return ErrorCode.RATE_LIMITED.value

    declared = getattr(exc, "code", None)
    if isinstance(declared, str) and declared in _BY_VALUE:
        return _BY_VALUE[declared].value

    name = type(exc).__name__
    if "Timeout" in name:
        return ErrorCode.TIMEOUT.value
    if name in {"ConnectError", "ReadError", "RemoteProtocolError", "ConnectTimeout"}:
        return ErrorCode.LLM_UNAVAILABLE.value
    if status is not None and 500 <= status < 600:
        return ErrorCode.LLM_UNAVAILABLE.value

    return ErrorCode.UNKNOWN.value


def normalise(code: str | None) -> str:
    """Coerce a stored code back into the vocabulary, defaulting to `unknown`.

    Rows written before this column existed carry an empty string; a row whose
    code was written by an older, wider vocabulary carries something not in the
    enum. Both are `unknown` for grouping purposes — neither is an error here.
    """
    if not code:
        return ErrorCode.UNKNOWN.value
    return _BY_VALUE.get(code, ErrorCode.UNKNOWN).value
