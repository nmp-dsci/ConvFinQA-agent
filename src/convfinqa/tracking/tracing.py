"""MLflow tracing: every LLM call the app makes, captured and joined to runs.

`mlflow.pydantic_ai.autolog()` records each agent invocation — and the chat
calls inside it — as spans in the tracking store's Traces tab. This module adds
the two levels the autologger cannot infer, the conversation (report) and the
turn (question), and stamps the run identity (version, split, run name) on the
trace so a span joins back to the experiment run that produced it.

Off unless something calls `enable()`: the eval loop always does (it already
requires MLflow), serving does when `MLFLOW_TRACING=1`. Importing this module
never imports mlflow — serving must stay cheap to start and the demo container
has no tracking server to reach.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("convfinqa.tracking")

_enabled = False


def enable() -> bool:
    """Turn on pydantic-ai autologging into the configured MLflow store."""
    global _enabled
    if _enabled:
        return True
    try:
        from convfinqa.tracking import mlflow_log

        mlflow_log._mlflow()  # point mlflow at the store and experiment
        import mlflow.pydantic_ai as _autolog_mod

        _autolog_mod.autolog()
    except Exception:  # noqa: BLE001 — tracing is never load-bearing
        log.warning("mlflow tracing unavailable; continuing without it", exc_info=True)
        return False
    _enabled = True
    return True


def enabled() -> bool:
    """Whether `enable()` has succeeded in this process."""
    return _enabled


class SpanHandle:
    """What a `span()` block yields: a way to add attributes once you know them.

    Some of the most useful attributes on a span are only knowable *after* the
    work inside it has run — a diagnosis span cannot carry the teacher's reason
    at the moment it opens. Attributes set through this handle land on the span
    that is still open, so the trace ends up describing the outcome rather than
    only the inputs. It is a no-op object when tracing is off, so a caller never
    has to check.
    """

    __slots__ = ("_span",)

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def set(self, **attributes: Any) -> None:
        """Add attributes to the open span, dropping any that are None."""
        if self._span is None:
            return
        with contextlib.suppress(Exception):
            for key, value in attributes.items():
                if value is not None:
                    self._span.set_attribute(key, value)


@contextlib.contextmanager
def span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    trace_tags: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    """A named span when tracing is on; a free no-op when it is off.

    `trace_tags` are set on the whole trace (not the span), so pass them on the
    outermost span only — they are what joins a trace to its run and version.

    Yields a `SpanHandle` so the caller can attach attributes it only learns
    from the work inside the block.
    """
    if not _enabled:
        yield SpanHandle()
        return
    try:
        import mlflow
    except Exception:  # noqa: BLE001
        yield SpanHandle()
        return
    with mlflow.start_span(name, attributes=attributes or {}) as active:
        if trace_tags:
            with contextlib.suppress(Exception):
                mlflow.update_current_trace(
                    tags={k: str(v) for k, v in trace_tags.items()}
                )
        yield SpanHandle(active)
