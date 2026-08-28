"""Assemble src/convfinqa/prompts/<variant>.py from base + per-agent JSONL rule stores.

Variant name is `settings.variant` (default `v3_1`). Override at runtime via
`--variant v3_2` on the CLI or `VARIANT=v3_2` in the env. The generated module
path is computed at assemble time so multiple variants coexist on disk.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from convfinqa.config import settings
from convfinqa.diagnosis.models import AgentName, Rule
from convfinqa.diagnosis.rules_store import AGENTS, all_rules
from convfinqa.prompts import load as load_prompts

_PROMPT_VAR_NAMES: dict[AgentName, str] = {
    "triage": "TRIAGE_PROMPT",
    "preprocess": "PREPROCESS_PROMPT",
    "retriever": "RETRIEVER_PROMPT",
    "calculator": "CALCULATOR_PROMPT",
}

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def variant_module_path(variant: str | None = None) -> Path:
    """Path to src/convfinqa/prompts/<variant>.py."""
    return _PROMPTS_DIR / f"{variant or settings.variant}.py"


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
    """Append each agent's promoted rules to its base prompt."""
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


def write_variant_module(
    prompts: dict[AgentName, str], variant: str | None = None
) -> Path:
    """Generate `prompts/<variant>.py` from assembled prompts. Never hand-edit it."""
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
    path = variant_module_path(variant)
    path.write_text("\n".join(parts))
    return path


def assemble_variant(*, base_version: str = "v2", variant: str | None = None) -> Path:
    """Assemble the variant module from base prompts + variant rules store.

    `base_version` is the *input* prompts module to overlay rules on top of
    (e.g. v2 to start a fresh v3_1 run; v3_1 to chain v3_2 on top of v3_1).
    `variant` is the *output* variant name (defaults to `settings.variant`).
    """
    v = variant or settings.variant
    base = load_prompts(base_version)
    rules_by_agent = all_rules()
    prompts = assemble_prompts(base, rules_by_agent)
    path = write_variant_module(prompts, variant=v)
    # Reload the generated module so callers can `prompts.load(variant)` immediately.
    try:
        importlib.import_module(f"convfinqa.prompts.{v}")
        importlib.reload(importlib.import_module(f"convfinqa.prompts.{v}"))
    except Exception:  # noqa: BLE001
        pass
    return path
