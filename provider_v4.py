"""
Multi-agent provider for ConvFinQA.

Pipeline per conversation turn
───────────────────────────────
 Agent 1 (Planner)          – classifies the question, decides which agents are
                               needed and what to ask them; no tools, no doc.
 Agent 2 (Direct Retriever) – extracts values from the document for
                               self-contained queries (no prior-Q&A context).
 Agent 3 (Context Retriever)– extracts values from the document for queries
                               whose meaning depends on prior Q&A.
 Agent 4 (Calculator)       – uses MCP calculator tools to compute the answer
                               from the values returned by Agents 2 / 3.
 Agent 5 (Reviewer)         – strips currency symbols, commas, extra text so
                               the final answer is a plain number or percentage.
"""

import asyncio
import json
import os
import re

import nest_asyncio
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompt_utils import golden_examples, chain_of_thought

nest_asyncio.apply()
load_dotenv()

client = Anthropic()
MODEL  = "claude-sonnet-4-6"

_HERE             = os.path.dirname(os.path.abspath(__file__))
PYTHON            = os.path.join(_HERE, ".venv/bin/python")
CALCULATOR_SERVER = os.path.join(_HERE, "mcp/server_calculator.py")


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Parse JSON from an LLM response, tolerating markdown fences."""
    # strip ```json … ``` fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    raise ValueError(f"Could not parse JSON from agent response: {text[:300]}")


def _history_str(conversation_history: list) -> str:
    if not conversation_history:
        return "None"
    return "\n".join(
        f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']}"
        for i, h in enumerate(conversation_history)
    )


# ── Agent 1 – Planner ────────────────────────────────────────────────────────

def agent_1_planner(question: str, conversation_history: list) -> tuple[dict, int, int]:
    """
    Classifies the question and produces a retrieval + calculation plan.
    Does NOT receive the document — classification is based on question
    phrasing and conversation history alone.

    Returns (plan_dict, input_tokens, output_tokens).

    plan_dict schema:
    {
      "turn_type": "numeric" | "calculation",
      "direct_queries": [...],   # self-contained doc-lookup questions
      "context_queries": [...],  # doc-lookup questions that need prior Q&A context
      "calculation_steps": "..."  # description of arithmetic to apply to retrieved values
    }
    """
    system = """\
You are a financial analysis planner. Your only job is to create a plan for
answering a question; you do not answer the question yourself.

Return ONLY valid JSON — no prose, no markdown fences — matching this schema:
{
  "turn_type": "numeric" | "calculation",
  "direct_queries": ["fully-specified question answerable from the document alone"],
  "context_queries": ["question whose meaning requires prior Q&A context"],
  "calculation_steps": "plain-English description of the arithmetic to perform, or empty string"
}

Definitions
-----------
turn_type "numeric"     : the answer is a number that can be read directly from
                          the document (no arithmetic needed beyond a simple lookup).
turn_type "calculation" : the answer requires arithmetic on one or more numbers.
direct_queries          : each query must be fully self-contained and answerable
                          from the financial document without knowing prior questions.
context_queries         : queries whose subject is implied by the conversation
                          (e.g. "what about in 2008?" needs context to know the topic).
calculation_steps       : describe the operation(s) to apply to the retrieved values
                          (e.g. "subtract 2008 value from 2009 value").
                          Leave empty string when turn_type is "numeric".\
"""
    user = f"""\
Prior conversation:
{_history_str(conversation_history)}

Current question: {question}

Produce the plan.\
"""
    resp = client.messages.create(
        model=MODEL, max_tokens=512, system=system,
        messages=[{"role": "user", "content": user}],
    )
    plan = _extract_json(resp.content[0].text)
    return plan, resp.usage.input_tokens, resp.usage.output_tokens


# ── Agent 2 – Direct Document Retriever ──────────────────────────────────────

def agent_2_direct_retriever(queries: list, doc: dict) -> tuple[dict, int, int]:
    """
    Extracts numeric values from the document for self-contained queries.
    No conversation history — each query must be fully specified.

    Returns (values_dict, input_tokens, output_tokens).
    values_dict maps each query string to its extracted numeric value string.
    """
    if not queries:
        return {}, 0, 0

    system = """\
You are a precise financial data extractor. Extract exact numeric values from
the financial document for each question listed.

Return ONLY valid JSON — no prose, no markdown fences:
{"question text": "raw_numeric_value", ...}

Extract the number exactly as it appears in the document (no formatting changes).
If a value is not found, use "NOT_FOUND".\
"""
    user = f"""\
Financial document:
{json.dumps(doc, indent=2)}

Questions to answer:
{json.dumps(queries, indent=2)}\
"""
    resp = client.messages.create(
        model=MODEL, max_tokens=512, system=system,
        messages=[{"role": "user", "content": user}],
    )
    values = _extract_json(resp.content[0].text)
    return values, resp.usage.input_tokens, resp.usage.output_tokens


# ── Agent 3 – Context-Aware Document Retriever ───────────────────────────────

def agent_3_context_retriever(
    queries: list, doc: dict, conversation_history: list
) -> tuple[dict, int, int]:
    """
    Extracts numeric values from the document for queries whose meaning
    requires prior conversation context (e.g. "what about in 2008?").

    Returns (values_dict, input_tokens, output_tokens).
    """
    if not queries:
        return {}, 0, 0

    system = """\
You are a precise financial data extractor. Some questions refer to values
whose subject is implied by the prior conversation. Use the conversation
history to resolve what each question is asking, then extract the exact
numeric value from the document.

Return ONLY valid JSON — no prose, no markdown fences:
{"question text": "raw_numeric_value", ...}

Extract the number exactly as it appears in the document (no formatting changes).
If a value is not found, use "NOT_FOUND".\
"""
    user = f"""\
Financial document:
{json.dumps(doc, indent=2)}

Prior conversation (use this to resolve ambiguous questions):
{_history_str(conversation_history)}

Questions to answer:
{json.dumps(queries, indent=2)}\
"""
    resp = client.messages.create(
        model=MODEL, max_tokens=512, system=system,
        messages=[{"role": "user", "content": user}],
    )
    values = _extract_json(resp.content[0].text)
    return values, resp.usage.input_tokens, resp.usage.output_tokens


# ── Agent 4 – Calculator ──────────────────────────────────────────────────────

async def agent_4_calculator(
    retrieved_values: dict,
    calculation_steps: str,
    session: ClientSession,
    tools: list,
) -> tuple[str, int, int]:
    """
    Performs the required arithmetic using MCP calculator tools.
    Must use tools for ALL arithmetic — never computes inline.

    Returns (result_string, input_tokens, output_tokens).
    """
    system = """\
You are a financial calculator. You have access to arithmetic tools.
You MUST use the tools for every calculation — never compute numbers yourself.
After the final tool call produces the answer, output ONLY that numeric result
— no explanation, no units, no currency symbols.\
"""
    user = f"""\
Retrieved values:
{json.dumps(retrieved_values, indent=2)}

Calculation required: {calculation_steps}

Use the calculator tools to compute and return the answer.\
"""
    messages = [{"role": "user", "content": user}]
    total_input = total_output = 0

    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=512, system=system,
            messages=messages, tools=tools,
        )
        total_input  += resp.usage.input_tokens
        total_output += resp.usage.output_tokens

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    mcp_result  = await session.call_tool(block.name, block.input)
                    result_text = (
                        mcp_result.content[0].text if mcp_result.content else "0"
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            text = next(
                (b.text for b in resp.content if hasattr(b, "text")), ""
            )
            return text.strip(), total_input, total_output


# ── Agent 5 – Reviewer ────────────────────────────────────────────────────────

def agent_5_reviewer(raw_answer: str, question: str) -> tuple[str, int, int]:
    """
    Cleans the final answer: removes currency symbols, commas, extra prose.
    Output is a plain number or percentage (e.g. "25587" or "14.14%").

    Returns (clean_answer, input_tokens, output_tokens).
    """
    system = """\
You are a financial answer formatter. Given a raw answer, return it in clean format.

Rules:
- Remove ALL currency symbols ($, £, €, etc.)
- Remove ALL commas from numbers (e.g. "206,588" → "206588")
- Keep the percentage sign if the answer is a percentage (e.g. "14.14%")
- Keep a leading minus sign for negative numbers
- If the answer is "yes" or "no", return it as-is
- Output ONLY the cleaned value — no explanation, no extra text\
"""
    user = f"""\
Question : {question}
Raw answer: {raw_answer}

Return the cleaned answer.\
"""
    resp = client.messages.create(
        model=MODEL, max_tokens=64, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def _run_conversation(doc: dict, turns: list) -> tuple[list, int, int]:
    """
    Runs the full multi-agent pipeline for every conversation turn.
    Opens a single MCP connection for the entire conversation.

    Returns (all_responses, total_input_tokens, total_output_tokens).
    """
    server_params = StdioServerParameters(command=PYTHON, args=[CALCULATOR_SERVER])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools_response.tools
            ]

            conversation_history: list[dict] = []
            all_responses: list[str] = []
            total_input = total_output = 0

            for turn in turns:
                question = turn["user"]

                # ── Agent 1: Plan ─────────────────────────────────────────
                plan, inp, out = agent_1_planner(question, conversation_history)
                total_input += inp; total_output += out

                turn_type        = plan.get("turn_type", "numeric")
                direct_queries   = plan.get("direct_queries", [])
                context_queries  = plan.get("context_queries", [])
                calculation_steps = plan.get("calculation_steps", "")

                # ── Agent 2: Direct retrieval ─────────────────────────────
                direct_values, inp, out = agent_2_direct_retriever(direct_queries, doc)
                total_input += inp; total_output += out

                # ── Agent 3: Context-aware retrieval ──────────────────────
                context_values, inp, out = agent_3_context_retriever(
                    context_queries, doc, conversation_history
                )
                total_input += inp; total_output += out

                all_values = {**direct_values, **context_values}

                # ── Agent 4: Calculate (only when needed) ─────────────────
                if turn_type == "calculation" and calculation_steps:
                    raw_answer, inp, out = await agent_4_calculator(
                        all_values, calculation_steps, session, tools
                    )
                    total_input += inp; total_output += out
                else:
                    # Numeric: take the single retrieved value
                    raw_answer = next(
                        (v for v in all_values.values() if v != "NOT_FOUND"),
                        "",
                    )

                # ── Agent 5: Review & clean ───────────────────────────────
                final_answer, inp, out = agent_5_reviewer(raw_answer, question)
                total_input += inp; total_output += out

                conversation_history.append({"question": question, "answer": final_answer})
                all_responses.append(final_answer)

            return all_responses, total_input, total_output


# ── promptfoo entry-point ─────────────────────────────────────────────────────

def call_api(prompt, options, context):
    """Called by promptfoo once per test case."""

    doc = context.get("vars", {}).get("doc", "")
    if isinstance(doc, str) and doc.startswith("{"):
        doc = json.loads(doc)

    turns = context.get("test", {}).get("conversation", [])
    if not turns:
        turns = [{"user": prompt}]

    loop = asyncio.get_event_loop()
    all_responses, total_input, total_output = loop.run_until_complete(
        _run_conversation(doc, turns)
    )

    return {
        "output": "\n\n---\n\n".join(all_responses),
        "tokenUsage": {
            "total":      total_input + total_output,
            "prompt":     total_input,
            "completion": total_output,
        },
    }
