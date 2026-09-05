"""Trim the repetitive bulk out of autologged spans before they are stored.

The pipeline's traces are 99.4% of the tracking store — 874 MB of 880 MB — and
almost none of it is information. Three span types account for it, and each is
bloated for a different reason:

- ``ToolManager.execute_tool_call`` (387 MB) serialises pydantic-ai's *toolset*
  beside the call: output schemas, JSON schemas, pydantic validators. Measured
  over 400 spans that is 4,909 B of framework noise against 817 B of actual
  call — 86% — and it is byte-identical on every tool call the agent ever makes.
- ``Agent.run`` (404 MB) writes ``_state`` and ``_new_messages_serialized``,
  ~112 KB each of internal run state, beside a 1.9 KB ``output``. The messages
  are already recorded properly by the model span below it, so this is a second
  copy of something kept elsewhere.
- ``InstrumentedModel.request`` (81 MB) is the honest one: the real request. Its
  bulk is the four agent system prompts, repeated on every call of every
  question. Those have an identity already — the prompt ledger's
  ``p2@4bc21f75`` — so the text is replaced by that reference and read back the
  same way `evalloop/prompt_refs` reads back a teacher prompt. Matching is on
  the **stripped** text: pydantic-ai delivers instructions with trailing
  whitespace removed, so hashing the raw module constant matches nothing and
  fails silently, leaving every prompt stored in full.

What is left afterwards is the model's own messages and outputs — the report
text, the questions, the tool arguments, the answers. That is deliberately kept:
it is the reasoning, and it is what a trace is for.

What survives is what a person actually reads a trace for: the question, the
tool calls and their arguments, the model's messages and reasoning, tokens and
latency. Nothing removed here is unrecoverable — the schemas come from the code,
the messages from the span below, the prompts from the ledger.

Set ``MLFLOW_TRACE_FULL=1`` to disable all of it when a raw trace is genuinely
needed for debugging.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

log = logging.getLogger("convfinqa.tracking")

# What replaces a dropped blob, so a reader sees a deliberate omission rather
# than wondering whether the field failed to serialise.
DROPPED = "<trimmed by convfinqa.tracking.span_trim — see that module>"


def enabled() -> bool:
    """Trimming is on unless someone asks for raw traces."""
    return os.environ.get("MLFLOW_TRACE_FULL", "") not in {"1", "true", "True"}


@lru_cache(maxsize=1)
def _prompt_index() -> dict[str, str]:
    """Hash of every registered agent prompt -> its ledger name, `preprocess p2`.

    Built from the committed bundle modules via the registry's lineages, so it
    covers every prompt version this repo has ever run rather than only the one
    in front of it.

    Keyed on the **stripped** text. pydantic-ai hands the model its instructions
    with trailing whitespace removed, so the observed prompt is one byte shorter
    than the module constant and hashing the raw text finds nothing — which
    fails silently, leaving every prompt stored in full. Stripping both sides is
    what makes the substitution actually fire.
    """
    out: dict[str, str] = {}
    try:
        import convfinqa.prompts as prompts_pkg
        from convfinqa.tracking import registry
        from convfinqa.tracking.prompt_ledger import prompt_hash

        doc = registry.load()
        by_hash = {
            str(e.get("hash", "")): f"{agent} {e.get('seq', '?')}"
            for agent, lineage in (doc.agent_prompts or {}).items()
            for e in lineage
        }
        for version in prompts_pkg.latest_all():
            for agent, text in prompts_pkg.load(version).items():
                h = prompt_hash(text)
                out[prompt_hash(text.strip())] = by_hash.get(h) or f"{agent} ?@{h}"
    except Exception:  # noqa: BLE001 — no index just means no substitution
        log.debug("prompt index unavailable; prompts stay inline", exc_info=True)
    return out


def _prompt_ref(text: str) -> str | None:
    """The ledger's name for this prompt text, if it is one we have registered."""
    if not text or len(text) < 200:
        return None
    try:
        from convfinqa.tracking.prompt_ledger import prompt_hash

        return _prompt_index().get(prompt_hash(text.strip()))
    except Exception:  # noqa: BLE001
        return None


def _trim_tool_call(value: Any) -> Any:
    """Keep the call; drop the toolset and the message history beside it.

    `ctx` carries a second copy of the whole conversation — the same messages
    the model span already records properly — which on a real span was 24 KB of
    the 26 KB. The small `ctx` keys (model, prompt, usage, run_id) stay: they
    are what makes the tool call attributable.
    """
    if not isinstance(value, dict):
        return value
    validated = value.get("validated")
    if not isinstance(validated, dict):
        return value
    trimmed = dict(validated)
    if "tool" in trimmed:
        trimmed["tool"] = DROPPED
    ctx = trimmed.get("ctx")
    if isinstance(ctx, dict):
        # `messages` is the whole conversation again; `prompt` is the user
        # prompt again. Both are verbatim copies of content the model span in
        # the same trace already carries — checked, not assumed: all 8 tool
        # calls of a sample conversation matched a model span exactly.
        drop = {k: DROPPED for k in ("messages", "prompt") if k in ctx}
        trimmed["ctx"] = {**ctx, **drop}
    value["validated"] = trimmed
    return value


def _trim_agent_run(value: Any) -> Any:
    """Keep the agent's output; drop the duplicated internal run state."""
    if not isinstance(value, dict):
        return value
    for key in ("_state", "_new_messages_serialized"):
        if key in value:
            value[key] = DROPPED
    return value


def _trim_model_request(value: Any) -> Any:
    """Replace registered system prompts with their ledger reference.

    pydantic-ai puts the system prompt on each message's ``instructions``, not
    in a part, and repeats it on every message of every request — so one agent
    prompt is stored tens of thousands of times across a run. Parts are checked
    too, since a prompt delivered as a system part should be substituted the
    same way.
    """
    if not isinstance(value, dict):
        return value
    messages = value.get("messages")
    if not isinstance(messages, list):
        return value
    for message in messages:
        if not isinstance(message, dict):
            continue
        ref = _prompt_ref(message.get("instructions") or "")
        if ref:
            message["instructions"] = f"<prompt_ref: {ref}>"
        for part in message.get("parts") or []:
            if not isinstance(part, dict):
                continue
            content = part.get("content")
            if not isinstance(content, str):
                continue
            part_ref = _prompt_ref(content)
            if part_ref:
                part["content"] = f"<prompt_ref: {part_ref}>"
    return value


TRIMMERS = {
    "ToolManager.execute_tool_call": ("inputs", _trim_tool_call),
    "Agent.run": ("outputs", _trim_agent_run),
    "InstrumentedModel.request": ("inputs", _trim_model_request),
}


def trim_span(span: Any) -> None:
    """MLflow span processor: shrink the known-repetitive payloads in place.

    Never raises. A span processor runs inside the tracing path on every span,
    so an exception here would turn an observability feature into a source of
    failures in the thing being observed — the opposite of the point.
    """
    if not enabled():
        return
    try:
        rule = TRIMMERS.get(getattr(span, "name", ""))
        if rule is None:
            return
        slot, trim = rule
        current = getattr(span, slot, None)
        if current is None:
            return
        # Round-trip through JSON so the trimmer works on plain data and never
        # mutates a live framework object the run is still using.
        try:
            data = json.loads(json.dumps(current, default=str))
        except Exception:  # noqa: BLE001
            return
        trimmed = trim(data)
        (span.set_inputs if slot == "inputs" else span.set_outputs)(trimmed)
    except Exception:  # noqa: BLE001
        log.debug(
            "span trimming failed for %r", getattr(span, "name", "?"), exc_info=True
        )
