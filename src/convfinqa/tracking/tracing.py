"""MLflow tracing: every LLM call the app makes, captured and joined to runs.

`mlflow.pydantic_ai.autolog()` records each agent invocation — and the chat
calls inside it — as spans in the tracking store's Traces tab. This module adds
the two levels the autologger cannot infer, the conversation (report) and the
turn (question), and stamps the run identity (version, split, run name) on the
trace so a span joins back to the experiment run that produced it.

**The Agent SDK gets none of that and must be instrumented by hand.** The four
pipeline agents run on pydantic-ai in this process, so autologging sees every
call. The teacher and the prompt writer do not: `claude_agent_sdk` spawns the
`claude` CLI as a *subprocess*, and no in-process client is ever constructed, so
there is nothing for any autologger to patch — and there is no
`mlflow.claude_agent_sdk` integration to enable. Without a manual span the whole
teacher call is invisible: a trace with one empty wrapper span, no prompt, no
reply, no tokens, no cost. `evalloop/sdk.py` opens that span around the single
chokepoint every Agent SDK call passes through; if a second call path to the SDK
is ever added, it needs the same treatment or it will silently record nothing.

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
        # The autologger writes what pydantic-ai hands it, which is mostly
        # framework internals: serialised toolsets, duplicated run state, and
        # the same four system prompts on every call. Trim them on the way out
        # — see `span_trim` for what goes and why each is recoverable.
        _install_trimmer()
    except Exception:  # noqa: BLE001 — tracing is never load-bearing
        log.warning("mlflow tracing unavailable; continuing without it", exc_info=True)
        return False
    _enabled = True
    return True


def _install_trimmer() -> None:
    """Register the span processor that drops the repetitive bulk."""
    try:
        import mlflow.tracing

        from convfinqa.tracking.span_trim import trim_span

        mlflow.tracing.configure(span_processors=[trim_span])
    except Exception:  # noqa: BLE001 — a fat trace beats no trace
        log.warning(
            "span trimming unavailable; traces will be full size", exc_info=True
        )


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

    def inputs(self, value: Any) -> None:
        """Record what went in. The trace UI leads with this; a span without it
        renders as an empty box whatever attributes it carries."""
        if self._span is None:
            return
        with contextlib.suppress(Exception):
            self._span.set_inputs(value)

    def outputs(self, value: Any) -> None:
        """Record what came back."""
        if self._span is None:
            return
        with contextlib.suppress(Exception):
            self._span.set_outputs(value)


@contextlib.contextmanager
def span(
    name: str,
    *,
    span_type: str = "UNKNOWN",
    attributes: dict[str, Any] | None = None,
    trace_tags: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    """A named span when tracing is on; a free no-op when it is off.

    `trace_tags` are set on the whole trace (not the span), so pass them on the
    outermost span only — they are what joins a trace to its run and version.

    `span_type` is one of MLflow's span types (`LLM`, `AGENT`, `TOOL`, `CHAIN`,
    …). It is worth setting: the UI groups and renders by type, and a span left
    `UNKNOWN` is shown as an anonymous box even when it is the model call the
    whole trace exists to record.

    Yields a `SpanHandle` so the caller can attach attributes, inputs and
    outputs it only learns from the work inside the block.
    """
    if not _enabled:
        yield SpanHandle()
        return
    try:
        import mlflow
    except Exception:  # noqa: BLE001
        yield SpanHandle()
        return
    with mlflow.start_span(
        name, span_type=span_type, attributes=attributes or {}
    ) as active:
        if trace_tags:
            with contextlib.suppress(Exception):
                mlflow.update_current_trace(
                    tags={k: str(v) for k, v in trace_tags.items()}
                )
        yield SpanHandle(active)
