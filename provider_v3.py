import asyncio
import json
import os

import nest_asyncio
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompt_utils import *

# Allow asyncio.run() inside an already-running event loop (needed when
# promptfoo or Jupyter already owns the loop).
nest_asyncio.apply()

load_dotenv()
client = Anthropic()

_HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(_HERE, ".venv/bin/python")
CALCULATOR_SERVER = os.path.join(_HERE, "mcp/server_calculator.py")


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------

async def _run_conversation(system_prompt: str, turns: list) -> tuple[list, int, int]:
    """
    Open one stdio connection to server_calculator.py, then run every
    conversation turn through Claude with a full agentic tool-use loop.

    Returns (all_responses, total_input_tokens, total_output_tokens).
    """
    server_params = StdioServerParameters(
        command=PYTHON,
        args=[CALCULATOR_SERVER],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch tool schemas from the MCP server
            tools_response = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools_response.tools
            ]

            messages = []
            all_responses = []
            total_input = 0
            total_output = 0

            for turn in turns:
                messages.append({"role": "user", "content": turn["user"]})

                # Agentic loop: keep going until end_turn (Claude may call
                # tools multiple times before giving a final answer).
                while True:
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        system=system_prompt,
                        messages=messages,
                        tools=tools,
                    )
                    total_input += response.usage.input_tokens
                    total_output += response.usage.output_tokens

                    if response.stop_reason == "tool_use":
                        # Append the assistant's tool-call message
                        messages.append({"role": "assistant", "content": response.content})

                        # Execute every tool call and collect results
                        tool_results = []
                        for block in response.content:
                            if block.type == "tool_use":
                                mcp_result = await session.call_tool(block.name, block.input)
                                result_text = (
                                    mcp_result.content[0].text
                                    if mcp_result.content
                                    else "0"
                                )
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_text,
                                })

                        # Feed results back as the next user message
                        messages.append({"role": "user", "content": tool_results})

                    else:
                        # end_turn (or unexpected stop) — extract the text answer
                        text = next(
                            (b.text for b in response.content if hasattr(b, "text")),
                            "",
                        )
                        messages.append({"role": "assistant", "content": response.content})
                        all_responses.append(text)
                        break

            return all_responses, total_input, total_output


# ---------------------------------------------------------------------------
# promptfoo entry-point
# ---------------------------------------------------------------------------

def call_api(prompt, options, context):
    """
    Called by promptfoo once per test case.  When the test uses the
    conversation: format all turns are in context.test.conversation;
    otherwise the single prompt is used.
    """

    # Load doc from context.vars (promptfoo resolves file:// refs)
    doc = context.get("vars", {}).get("doc", "")
    if isinstance(doc, str) and doc.startswith("{"):
        doc = json.loads(doc)

    system_prompt = f"""You are a financial analyst assistant.
Use the following financial document to answer every question accurately.
Maintain full context from earlier questions in the conversation.

<document>
{json.dumps(doc, indent=2)}
</document>

<examples>
{golden_examples}
</examples>

<chain_of_thought>
{chain_of_thought}
</chain_of_thought>

<calculator_instructions>
You have access to calculator tools: add, subtract, multiply, divide, exp, greater.
You MUST use these tools for ALL numerical calculations — never compute numbers
yourself.  Even simple arithmetic like 206588 - 181001 must go through the tool.
After the tool returns the result, use that exact number in your answer.
</calculator_instructions>

Only output the final answer value, no explanation."""

    turns = context.get("test", {}).get("conversation", [])

    # Single-turn fallback (e.g. called directly from run_provider.py)
    if not turns:
        turns = [{"user": prompt}]

    loop = asyncio.get_event_loop()
    all_responses, total_input, total_output = loop.run_until_complete(
        _run_conversation(system_prompt, turns)
    )

    combined_output = "\n\n---\n\n".join(all_responses)

    return {
        "output": combined_output,
        "tokenUsage": {
            "total": total_input + total_output,
            "prompt": total_input,
            "completion": total_output,
        },
    }
