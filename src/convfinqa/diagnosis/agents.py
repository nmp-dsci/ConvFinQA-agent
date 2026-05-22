"""Five LLM-backed agents for the s7 diagnosis harness (router + 4 specialists)."""

from __future__ import annotations

import re

from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput

from convfinqa.backends.pydantic import LM_MAX
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

diagnostic_router_agent = Agent(
    LM_MAX,
    output_type=PromptedOutput(RouterDiagnosis),
    instructions=DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
)
triage_fix_agent = Agent(
    LM_MAX,
    output_type=PromptedOutput(FixProposal),
    instructions=FIX_TRIAGE_SYSTEM_PROMPT,
)
preprocess_fix_agent = Agent(
    LM_MAX,
    output_type=PromptedOutput(FixProposal),
    instructions=FIX_PREPROCESS_SYSTEM_PROMPT,
)
retriever_fix_agent = Agent(
    LM_MAX,
    output_type=PromptedOutput(FixProposal),
    instructions=FIX_RETRIEVER_SYSTEM_PROMPT,
)
calculator_fix_agent = Agent(
    LM_MAX,
    output_type=PromptedOutput(FixProposal),
    instructions=FIX_CALCULATOR_SYSTEM_PROMPT,
)

FIX_AGENTS: dict[str, Agent] = {
    "triage": triage_fix_agent,
    "preprocess": preprocess_fix_agent,
    "retriever": retriever_fix_agent,
    "calculator": calculator_fix_agent,
}

_FORBIDDEN_TOKENS = re.compile(
    r"\b(def\s|import\s|class\s|Agent\(|model\s*=|temperature\s*=|tools\s*=|pipeline)\b"
)


async def route_case(payload: RouterPayload) -> RouterDiagnosis:
    """Step 1 — Diagnose: classify-only. One LM_MAX call per case."""
    result = await diagnostic_router_agent.run(payload.model_dump_json())
    return result.output


async def propose_fix(failed_agent: str, payload: FixPayload) -> FixProposal:
    """Step 2 — Route+Fix: dispatch to the specialist for failed_agent."""
    if failed_agent not in FIX_AGENTS:
        raise ValueError(
            f"propose_fix called with unknown failed_agent={failed_agent!r}; "
            f"expected one of {list(FIX_AGENTS)}"
        )
    result = await FIX_AGENTS[failed_agent].run(payload.model_dump_json())
    proposal = result.output
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
