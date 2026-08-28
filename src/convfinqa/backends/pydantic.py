"""Pydantic AI backend: the four pipeline agents, built lazily.

Agents are constructed on first use, not at import. Two reasons, both
load-bearing: constructing a model demands an API key, and the keyless clone /
demo container must still import this module; and building them lazily is what
lets `DEMO_MODE` refuse at the choke point rather than at import, where the
failure would be a stack trace at startup instead of a typed 501.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

from convfinqa.config import settings
from convfinqa.llm import LM_MINI_MODEL, get_model
from convfinqa.pipeline.prompts_loader import PROMPTS
from convfinqa.pipeline.stages import CalcOut, PreprocessOut, RetrievedValues, TriageOut
from convfinqa.pipeline.tools import CALCULATOR_TOOLS

# Re-exported for callers that name the models explicitly (the s7 harness picks
# LM_MAX for its router and fix agents).
LM_MINI_NAME = LM_MINI_MODEL


def lm_mini() -> Any:
    """The fast model the four pipeline agents run on, every turn."""
    return get_model(LM_MINI_MODEL)


def lm_max() -> Any:
    """The flagship model used where reasoning quality is the product (s7)."""
    return get_model(settings.lm_max_model)


def make_agents(version_prompts: dict[str, str]) -> dict[str, Agent[None, Any]]:
    """Build a fresh set of four pipeline agents from a prompts dict."""
    model = lm_mini()
    calc: Agent[None, Any] = Agent(
        model, output_type=CalcOut, instructions=version_prompts["calculator"]
    )
    for fn in CALCULATOR_TOOLS:
        calc.tool_plain(fn)
    return {
        "triage": Agent(
            model, output_type=TriageOut, instructions=version_prompts["triage"]
        ),
        "preprocess": Agent(
            model,
            output_type=PreprocessOut,
            instructions=version_prompts["preprocess"],
        ),
        "retriever": Agent(
            model,
            output_type=RetrievedValues,
            instructions=version_prompts["retriever"],
        ),
        "calculator": calc,
    }


_default_agents: dict[str, Agent[None, Any]] | None = None


def default_agents() -> dict[str, Agent[None, Any]]:
    """The four agents for the currently-resolved prompt bundle, built once."""
    global _default_agents
    if _default_agents is None:
        _default_agents = make_agents(PROMPTS)
    return _default_agents


def reset_default_agents() -> None:
    """Drop the cached agents. For tests that swap prompts or settings."""
    global _default_agents
    _default_agents = None


_make_agents = make_agents
