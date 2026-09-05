"""Read-only MLflow tools the teacher may call, as an in-process MCP server.

The baked ledger is always injected, so every teacher call has the same
guaranteed context and "what did the writer know when it wrote this?" has a
fixed answer. These tools are the other half of that decision: for the unusual
case, the agent can go and read more — a specific past prompt, the failures
attributed to an agent in a given version, the full attempt history.

Every tool is read-only by construction. There is no write path here at all, so
a teacher that misbehaves can waste tokens but cannot alter the record it is
being judged against. Tool calls appear in the message stream and are counted
onto the propose run, so the extra reads are logged rather than invisible.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

MAX_CHARS = 6000


def _text(payload: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + f"\n… truncated at {MAX_CHARS} characters"
    return {"content": [{"type": "text", "text": body}]}


@tool(
    "search_attempts",
    "Past prompt rewrites of one subagent with their gate outcomes, newest first.",
    {"target_agent": str, "limit": int},
)
async def search_attempts(args: dict[str, Any]) -> dict[str, Any]:
    """Attempt history for one agent — what was changed and whether it promoted."""
    from convfinqa.evalloop import ledger

    rows = ledger.attempts(
        target_agent=str(args.get("target_agent") or "") or None,
        limit=int(args.get("limit") or 10),
    )
    return _text([{k: v for k, v in r.items() if k != "prompt"} for r in rows])


@tool(
    "get_prompt",
    "The exact system prompt one subagent used in one bundle version.",
    {"version": str, "agent": str},
)
async def get_prompt(args: dict[str, Any]) -> dict[str, Any]:
    """One agent's prompt text from a registered version."""
    import convfinqa.prompts as prompts_pkg

    try:
        prompts = prompts_pkg.load(str(args["version"]))
    except Exception as exc:  # noqa: BLE001 — the agent should see why, not crash
        return _text(f"no such version {args.get('version')!r}: {exc}")
    agent = str(args.get("agent") or "")
    if agent not in prompts:
        return _text(f"unknown agent {agent!r}; have {sorted(prompts)}")
    return _text(prompts[agent])


@tool(
    "get_failures",
    "Diagnosed first-fault failures attributed to one subagent in one version.",
    {"version": str, "agent": str, "limit": int},
)
async def get_failures(args: dict[str, Any]) -> dict[str, Any]:
    """Past diagnoses for one agent — the concrete cases behind a fault count."""
    from convfinqa.evalloop.teacher import DIAGNOSTICS_DIR

    version = str(args.get("version") or "")
    agent = str(args.get("agent") or "")
    limit = int(args.get("limit") or 10)
    rows: list[dict[str, Any]] = []
    for path in sorted(
        DIAGNOSTICS_DIR.glob(f"diagnoses_{version}_*.jsonl"), reverse=True
    ):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if agent and d.get("failed_agent") != agent:
                continue
            rows.append(
                {
                    "report_id": d.get("report_id"),
                    "failure_mode": d.get("failure_mode"),
                    "what_went_wrong": d.get("what_went_wrong"),
                    "proposed_rule": d.get("proposed_rule"),
                }
            )
            if len(rows) >= limit:
                return _text(rows)
    return _text(rows or f"no diagnoses recorded for {agent!r} in {version!r}")


def loop_server() -> Any:
    """The MCP server handed to the prompt writer."""
    return create_sdk_mcp_server(
        name="loop",
        version="1.0.0",
        tools=[search_attempts, get_prompt, get_failures],
    )


ALLOWED_TOOLS = [
    "mcp__loop__search_attempts",
    "mcp__loop__get_prompt",
    "mcp__loop__get_failures",
]
