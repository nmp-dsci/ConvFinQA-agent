"""Per-agent JSONL stores for rules + rule_attempts (v3_opt)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from convfinqa.config import settings
from convfinqa.diagnosis.models import (
    AgentName,
    FailureReason,
    FixType,
    Rule,
    RuleAttempt,
    VerifyResult,
)

AGENTS: tuple[AgentName, ...] = ("triage", "preprocess", "retriever", "calculator")
_SUFFIX = "_v3_opt"


def rules_path(agent: AgentName) -> Path:
    return Path(settings.rules_dir) / f"rules_{agent}{_SUFFIX}.jsonl"


def attempts_path(agent: AgentName) -> Path:
    return Path(settings.rules_dir) / f"rule_attempts_{agent}{_SUFFIX}.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _short_uuid() -> str:
    return uuid.uuid4().hex[:6]


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_rules(agent: AgentName) -> list[Rule]:
    """Return active rules (filter out any rule_id referenced in a supersedes list)."""
    raw = _read_lines(rules_path(agent))
    superseded: set[str] = set()
    for entry in raw:
        superseded.update(entry.get("supersedes") or [])
    out: list[Rule] = []
    for entry in raw:
        if entry.get("rule_id") in superseded:
            continue
        try:
            out.append(Rule.model_validate(entry))
        except Exception:  # noqa: BLE001
            continue
    return out


def read_attempts(agent: AgentName, *, limit: int | None = None) -> list[RuleAttempt]:
    raw = _read_lines(attempts_path(agent))
    out: list[RuleAttempt] = []
    for entry in raw:
        try:
            out.append(RuleAttempt.model_validate(entry))
        except Exception:  # noqa: BLE001
            continue
    if limit is None:
        limit = settings.max_prior_attempts_in_payload
    if limit and len(out) > limit:
        out = out[-limit:]
    return out


def append_rule(
    agent: AgentName,
    rule_text: str,
    fix_type: FixType,
    confidence: float,
    report_id: str,
    turn_index: int,
    *,
    supersedes: list[str] | None = None,
) -> str:
    """Append a verified rule. Returns the new rule_id."""
    rule_id = f"{agent[:4]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_short_uuid()}"
    entry = Rule(
        rule_id=rule_id,
        agent=agent,
        rule=rule_text,
        fix_type=fix_type,
        confidence=confidence,
        verified_on=[{"report_id": report_id, "turn_index": turn_index}],
        verified_at=_now_iso(),
        supersedes=supersedes or [],
    )
    path = rules_path(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry.model_dump_json() + "\n")
    return rule_id


def append_attempt(
    agent: AgentName,
    rule_text: str,
    fix_type: FixType,
    confidence: float,
    report_id: str,
    turn_index: int,
    verify_result: VerifyResult,
    *,
    first_failing_turn: int | None = None,
    failure_reason: FailureReason | None = None,
    promoted_rule_id: str | None = None,
) -> str:
    """Append a rule attempt (pass or fail). Returns attempt_id."""
    attempt_id = (
        f"{agent[:4]}-att-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_short_uuid()}"
    )
    entry = RuleAttempt(
        attempt_id=attempt_id,
        agent=agent,
        rule=rule_text,
        fix_type=fix_type,
        confidence=confidence,
        verify_result=verify_result,
        attempted_on={"report_id": report_id, "turn_index": turn_index},
        attempted_at=_now_iso(),
        first_failing_turn=first_failing_turn,
        failure_reason=failure_reason,
        promoted_rule_id=promoted_rule_id,
    )
    path = attempts_path(agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry.model_dump_json() + "\n")
    return attempt_id


def reset_rules(agent: AgentName | None = None) -> None:
    """Truncate the rules + attempts stores. None = all agents."""
    targets = [agent] if agent else list(AGENTS)
    for a in targets:
        if rules_path(a).exists():
            rules_path(a).write_text("")
        if attempts_path(a).exists():
            attempts_path(a).write_text("")


def all_rules() -> dict[AgentName, list[Rule]]:
    return {a: read_rules(a) for a in AGENTS}
