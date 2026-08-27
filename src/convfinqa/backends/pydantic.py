"""Pydantic AI backend: LM provider + four pipeline agents."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from convfinqa.config import settings
from convfinqa.pipeline.prompts_loader import PROMPTS
from convfinqa.pipeline.stages import CalcOut, PreprocessOut, RetrievedValues, TriageOut
from convfinqa.pipeline.tools import CALCULATOR_TOOLS

_deepseek_provider = OpenAIProvider(
    base_url="https://api.deepseek.com/v1",
    api_key=settings.deepseek_api_key.get_secret_value(),
)
# DeepSeek model identifiers — migrated to v4 names ahead of the 2026-07-24
# deprecation of the legacy `deepseek-chat` / `deepseek-reasoner` aliases.
# - LM_MINI = `deepseek-v4-flash` (was `deepseek-chat`): 284B/13B MoE, fast,
#   used by the four production sub-agents on every turn.
# - LM_MAX  = `deepseek-v4-pro` by default via settings (was `deepseek-reasoner`):
#   1.6T/49B MoE, slower + more expensive, used by the s7 harness's diagnostic
#   router + four specialist Fix agents where reasoning quality matters most.
LM_MINI = OpenAIChatModel("deepseek-v4-flash", provider=_deepseek_provider)
LM_MAX = OpenAIChatModel(settings.lm_max_model, provider=_deepseek_provider)

triage_agent = Agent(LM_MINI, output_type=TriageOut, instructions=PROMPTS["triage"])
preprocess_agent = Agent(
    LM_MINI, output_type=PreprocessOut, instructions=PROMPTS["preprocess"]
)
retriever_agent = Agent(
    LM_MINI, output_type=RetrievedValues, instructions=PROMPTS["retriever"]
)
calculator_agent = Agent(
    LM_MINI, output_type=CalcOut, instructions=PROMPTS["calculator"]
)

for _fn in CALCULATOR_TOOLS:
    calculator_agent.tool_plain(_fn)


def make_agents(version_prompts: dict[str, str]) -> dict[str, Agent]:
    """Build a fresh set of four pipeline agents from a prompts dict."""
    calc = Agent(LM_MINI, output_type=CalcOut, instructions=version_prompts["calculator"])
    for fn in CALCULATOR_TOOLS:
        calc.tool_plain(fn)
    return {
        "triage": Agent(LM_MINI, output_type=TriageOut, instructions=version_prompts["triage"]),
        "preprocess": Agent(
            LM_MINI, output_type=PreprocessOut, instructions=version_prompts["preprocess"]
        ),
        "retriever": Agent(
            LM_MINI, output_type=RetrievedValues, instructions=version_prompts["retriever"]
        ),
        "calculator": calc,
    }


_make_agents = make_agents
