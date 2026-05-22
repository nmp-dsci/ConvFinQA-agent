"""Assemble src/convfinqa/prompts/v3_opt.py from v2 + per-agent JSONL rule stores."""

from __future__ import annotations

import importlib
from pathlib import Path

from convfinqa.diagnosis.models import AgentName, Rule
from convfinqa.diagnosis.rules_store import AGENTS, all_rules
from convfinqa.prompts import load as load_prompts

_PROMPT_VAR_NAMES: dict[AgentName, str] = {
    "triage": "TRIAGE_PROMPT",
    "preprocess": "PREPROCESS_PROMPT",
    "retriever": "RETRIEVER_PROMPT",
    "calculator": "CALCULATOR_PROMPT",
}

V3_OPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "v3_opt.py"


def _format_rules_block(rules: list[Rule]) -> str:
    if not rules:
        return ""
    lines = ["", "## Additional Rules (automated patch)", ""]
    for i, r in enumerate(rules, start=1):
        lines.append(f"{i}. ({r.rule_id}) {r.rule.strip()}")
    return "\n".join(lines) + "\n"


def assemble_prompts(
    base: dict[str, str], rules_by_agent: dict[AgentName, list[Rule]]
) -> dict[AgentName, str]:
    out: dict[AgentName, str] = {}
    for agent in AGENTS:
        base_prompt = base[agent].rstrip()
        block = _format_rules_block(rules_by_agent.get(agent, []))
        if block:
            out[agent] = base_prompt + "\n" + block
        else:
            out[agent] = base_prompt + "\n"
    return out


def _escape_triple_quotes(s: str) -> str:
    # Defensive: v2 doesn't currently use triple-quotes inside the prompt
    # body, but escape them just in case.
    return s.replace('"""', '\\"\\"\\"')


def write_v3_opt_module(prompts: dict[AgentName, str]) -> Path:
    parts = [
        '"""GENERATED — assembled by convfinqa.diagnosis.assembler. Do not hand-edit."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    for agent in AGENTS:
        var = _PROMPT_VAR_NAMES[agent]
        body = _escape_triple_quotes(prompts[agent])
        parts.append(f'{var} = """\\')
        parts.append(body.rstrip("\n"))
        parts.append('"""')
        parts.append("")
    V3_OPT_PATH.write_text("\n".join(parts))
    return V3_OPT_PATH


def assemble_v3_opt(*, base_version: str = "v2") -> Path:
    base = load_prompts(base_version)
    rules_by_agent = all_rules()
    prompts = assemble_prompts(base, rules_by_agent)
    path = write_v3_opt_module(prompts)
    # Reload the generated module so callers can `prompts.load("v3_opt")` immediately.
    try:
        importlib.import_module("convfinqa.prompts.v3_opt")
        importlib.reload(importlib.import_module("convfinqa.prompts.v3_opt"))
    except Exception:  # noqa: BLE001
        pass
    return path
