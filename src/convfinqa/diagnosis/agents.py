"""Five LLM-backed agents for the s7 diagnosis harness (router + 4 specialists)."""

from __future__ import annotations

import re
from functools import cache
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from convfinqa.backends.pydantic import lm_max
from convfinqa.diagnosis.models import (
    FixPayload,
    FixProposal,
    RouterDiagnosis,
    RouterPayload,
)
from convfinqa.diagnosis.prompts import (
    DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
    FIX_CALCULATOR_SYSTEM_PROMPT,
    FIX_PREPROCESS_SYSTEM_PROMPT,
    FIX_RETRIEVER_SYSTEM_PROMPT,
    FIX_TRIAGE_SYSTEM_PROMPT,
)

# Built lazily, like the pipeline agents: constructing a model demands a key,
# and the s7 module is imported by the CLI and the admin API in processes that
# may never run a round. `functools.cache` keeps it to one construction.


@cache
def _agents() -> dict[str, Agent[None, Any]]:
    """Router + four specialist Fix agents, all on the flagship model."""
    model = lm_max()
    return {
        "router": Agent(
            model,
            output_type=PromptedOutput(RouterDiagnosis),
            instructions=DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
        ),
        "triage": Agent(
            model,
            output_type=PromptedOutput(FixProposal),
            instructions=FIX_TRIAGE_SYSTEM_PROMPT,
        ),
        "preprocess": Agent(
            model,
            output_type=PromptedOutput(FixProposal),
            instructions=FIX_PREPROCESS_SYSTEM_PROMPT,
        ),
        "retriever": Agent(
            model,
            output_type=PromptedOutput(FixProposal),
            instructions=FIX_RETRIEVER_SYSTEM_PROMPT,
        ),
        "calculator": Agent(
            model,
            output_type=PromptedOutput(FixProposal),
            instructions=FIX_CALCULATOR_SYSTEM_PROMPT,
        ),
    }


FIX_AGENT_NAMES = ("triage", "preprocess", "retriever", "calculator")


_FORBIDDEN_TOKENS = re.compile(
    r"\b(def\s|import\s|class\s|Agent\(|model\s*=|temperature\s*=|tools\s*=|pipeline)\b"
)


async def route_case(payload: RouterPayload) -> RouterDiagnosis:
    """Step 1 — Diagnose: classify-only. One flagship-model call per case."""
    result = await _agents()["router"].run(payload.model_dump_json())
    diagnosis: RouterDiagnosis = result.output
    return diagnosis


async def propose_fix(failed_agent: str, payload: FixPayload) -> FixProposal:
    """Step 2 — Route+Fix: dispatch to the specialist for failed_agent."""
    if failed_agent not in FIX_AGENT_NAMES:
        raise ValueError(
            f"propose_fix called with unknown failed_agent={failed_agent!r}; "
            f"expected one of {list(FIX_AGENT_NAMES)}"
        )
    result = await _agents()[failed_agent].run(payload.model_dump_json())
    proposal: FixProposal = result.output
    if _FORBIDDEN_TOKENS.search(proposal.rule):
        # Hard-constraint guard: scrub the rule and downgrade confidence so
        # the harness routes the case to unresolved.
        proposal = FixProposal(
            rule="",
            fix_type=proposal.fix_type,
            confidence=0.0,
            rationale=(
                "rejected: proposed rule contained forbidden tokens "
                "(code-like content). Original rationale: " + proposal.rationale
            ),
        )
    return proposal
