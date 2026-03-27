"""
Multi-agent orchestrator for ConvFinQA  (provider_v5)
======================================================

Pipeline per conversation turn
───────────────────────────────

  planner_agent        Central orchestrator. Classifies the question into
                       turn_type ("Number" | "Program") and type2_question
                       (True | False).  When type2 is False it generates
                       the document lookup questions itself.  After retrieval
                       it also produces the ordered calculation steps for
                       Program questions.

  type2_sub_agent      Invoked only when type2_question=True.  Takes the
                       current question + full conversation history and
                       returns fully-specified document lookup questions
                       (resolving implicit references like "subsequent year"
                       or "that period" from earlier turns).

  document_sub_agent   Receives a list of self-contained lookup questions
                       and returns the exact numeric values from the document.

  program_sub_agent    Invoked only when turn_type="Program".  Receives the
                       retrieved values + an ordered list of calculation steps
                       and executes them using MCP calculator tools.

  review_sub_agent     Receives the raw final answer and returns a clean
                       number or percentage — no commas, no currency symbols,
                       no explanation.

Decision tree
─────────────
  type2_question=True  → type2_sub_agent  → number_questions
  type2_question=False → planner generates number_questions

  number_questions → document_sub_agent → retrieved_values

  turn_type="Number"  → retrieved_values[0] (single lookup, no calc)
  turn_type="Program" → planner builds calc_plan
                      → program_sub_agent (MCP tools) → raw_answer

  raw_answer → review_sub_agent → final_answer
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


# ── shared helpers ────────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Parse JSON from an LLM response, tolerating markdown fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"Could not parse JSON from agent response:\n{text[:400]}")


def _history_str(history: list) -> str:
    """Format conversation history as a readable string."""
    if not history:
        return "None"
    return "\n".join(
        f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']}"
        for i, h in enumerate(history)
    )


def _call(system: str, user: str, max_tokens: int = 512) -> tuple:
    """Single Claude call; returns (text, input_tokens, output_tokens)."""
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


# ── Agent: Planner (Phase 1 — classify + generate number_questions) ───────────

def planner_classify(question: str, history: list) -> tuple:
    """
    Classifies the question and, when type2_question=False, generates the
    document lookup questions needed.

    Returns
    -------
    plan : dict with keys
        turn_type         "Number" | "Program"
        type2_question    true | false
        number_questions  list of fully-specified lookup questions
                          (populated only when type2_question=false)
    inp, out : token counts
    """
    system = f"""\
You are a financial analysis orchestrator.  Your job is to classify an
incoming question and produce a lookup plan — you do NOT answer the question.

Definitions
-----------
turn_type "Number"
  The answer is read directly from the financial document; no arithmetic.
  Examples:
    * "what is the total revenue in 2019?"
    * "what is total net assets in 2008?"

turn_type "Program"
  The answer requires arithmetic on one or more numbers (add / subtract /
  multiply / divide / exp / greater).  The inputs come from the document
  and/or prior assistant answers.
  Examples:
    * "what is the percentage change from 2008 to 2009?"
    * "what is the total of those two years combined?"

type2_question true
  The current question is a re-framing or extension of a prior question that
  requires reading the prior question text to know what to look up.
  Examples:
    * Prior: "from 2018 to 2019, what was the change in total rental expense?"
      Current: "and over the subsequent year of that period, what was that change?"
      -> needs history to know this means 2019-to-2020 rental expense change
    * Prior: "what was the change in average price per share from 2012 to 2013?"
      Current: "and in the last year of that period, what was the total amount spent?"
      -> needs history to know what "that period" refers to

type2_question false
  The question can be planned without examining the prior question text (though
  it may still use prior answers as inputs to a Program).

number_questions
  Self-contained document lookup questions (each fully specified, no pronouns
  or implicit references). Leave as empty list [] when type2_question=true
  (the type2_sub_agent will generate these instead).

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "turn_type": "Number",
  "type2_question": false,
  "number_questions": ["fully-specified lookup question", ...]
}}

Golden examples for context:
{golden_examples}
"""
    user = f"""\
Prior conversation:
{_history_str(history)}

Current question: {question}

Produce the plan.\
"""
    text, inp, out = _call(system, user)
    plan = _extract_json(text)
    return plan, inp, out


# ── Agent: Type2 Sub-Agent ────────────────────────────────────────────────────

def type2_sub_agent(question: str, history: list) -> tuple:
    """
    Handles type2 questions by interpreting the current question in the
    context of prior questions and returning fully-specified document lookup
    questions.

    For example, if history contains "from 2018 to 2019, what was the change
    in total rental expense?" and the current question is "and over the
    subsequent year, what was that change?", this agent returns:
      ["What is the total rental expense in 2019?",
       "What is the total rental expense in 2020?"]

    Returns
    -------
    number_questions : list[str]
    inp, out : token counts
    """
    system = """\
You are a financial question interpreter.  You receive a question that
implicitly refers to a prior question in the conversation history.  Your
job is to resolve the implicit reference and return the fully-specified
document lookup questions needed to answer the current question.

Rules
-----
* Read the conversation history carefully to understand what the prior
  question was asking about (topic, metric, time period, entity, etc.).
* Apply any modification the current question makes (e.g. "subsequent year",
  "last year of that period", "the other year", "same metric for X").
* Each returned question must be independently answerable from the financial
  document with no implicit references or pronouns.
* Do NOT include arithmetic in the questions — only data lookups.

Return ONLY valid JSON — a list of strings, no markdown fences:
["fully-specified question 1", "fully-specified question 2", ...]
"""
    user = f"""\
Conversation history:
{_history_str(history)}

Current question (type2): {question}

Return the fully-specified document lookup questions.\
"""
    text, inp, out = _call(system, user)
    questions = _extract_json(text)
    if isinstance(questions, dict):
        questions = questions.get("number_questions", list(questions.values()))
    return questions, inp, out


# ── Agent: Document Sub-Agent ─────────────────────────────────────────────────

def document_sub_agent(number_questions: list, doc: dict) -> tuple:
    """
    Retrieves exact numeric values from the financial document for each
    lookup question.  Questions must be fully self-contained.

    Returns
    -------
    values : dict mapping question -> numeric value string
    inp, out : token counts
    """
    if not number_questions:
        return {}, 0, 0

    system = """\
You are a precise financial data extractor.

Given a financial document and a list of lookup questions, extract the exact
numeric value for each question directly from the document.

Rules
-----
* Return the number exactly as it appears (do not reformat or round).
* Do NOT perform any arithmetic.
* If a value is not found, use the string "NOT_FOUND".
* Return ONLY valid JSON — no prose, no markdown fences:
  {"question text": "value", ...}
"""
    user = f"""\
Financial document:
{json.dumps(doc, indent=2)}

Lookup questions:
{json.dumps(number_questions, indent=2)}\
"""
    text, inp, out = _call(system, user, max_tokens=1024)
    values = _extract_json(text)
    return values, inp, out


# ── Agent: Planner (Phase 2 — build calc plan for Program questions) ──────────

def planner_program(
    question: str,
    retrieved_values: dict,
    history: list,
) -> tuple:
    """
    For Program questions: given the values retrieved from the document and
    the conversation history, produces:
      * all_values  — merged dict of document values + any needed prior answers
      * calculation — ordered plain-English steps describing the arithmetic

    Returns
    -------
    calc_plan : dict with keys
        all_values   : {"label": "numeric_string", ...}
        calculation  : "ordered description of arithmetic steps"
    inp, out : token counts
    """
    system = """\
You are a financial calculation planner.  You receive retrieved document
values, the conversation history (which may contain prior answers needed
as inputs), and the current question that requires arithmetic.

Your job is to:
1. Collect ALL numeric inputs needed — from retrieved_values and/or prior
   assistant answers in the conversation history.
2. Describe the exact ordered sequence of arithmetic steps to compute the
   answer using: add, subtract, multiply, divide, exp, greater.

Return ONLY valid JSON — no prose, no markdown fences:
{
  "all_values": {
    "descriptive_label": "numeric_string_value"
  },
  "calculation": "Step-by-step plain-English description of the arithmetic,
                  referencing the labels in all_values."
}

Important
---------
* Labels in all_values must be descriptive (e.g. "net_cash_2009", "prior_answer_q1").
* Do NOT perform arithmetic yourself — only describe the steps.
* Only include values actually needed for the calculation.
* For percentage change: subtract base from new to get difference, then
  divide difference by base, then multiply by 100.
"""
    user = f"""\
Prior conversation (may contain inputs needed for calculation):
{_history_str(history)}

Values retrieved from the document:
{json.dumps(retrieved_values, indent=2)}

Current question requiring calculation: {question}

Produce the calculation plan.\
"""
    text, inp, out = _call(system, user)
    calc_plan = _extract_json(text)
    return calc_plan, inp, out


# ── Agent: Program Sub-Agent (MCP calculator) ─────────────────────────────────

async def program_sub_agent(
    all_values: dict,
    calculation: str,
    session: ClientSession,
    tools: list,
) -> tuple:
    """
    Executes the calculation using MCP calculator tools.
    Runs an agentic tool-use loop until Claude reaches end_turn.

    Returns
    -------
    result : computed numeric string
    inp, out : token counts
    """
    system = """\
You are a financial calculator.  You have access to arithmetic tools:
add, subtract, multiply, divide, exp, greater.

Rules
-----
* You MUST use a tool for EVERY arithmetic operation — never compute inline.
* Execute the steps in the order given.
* After the final tool call returns, output ONLY the numeric result —
  no explanation, no units, no currency symbols.
"""
    user = f"""\
Input values:
{json.dumps(all_values, indent=2)}

Calculation steps:
{calculation}

Execute the steps using the tools and return the final answer.\
"""
    messages = [{"role": "user", "content": user}]
    total_inp = total_out = 0

    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=512, system=system,
            messages=messages, tools=tools,
        )
        total_inp += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

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
            return text.strip(), total_inp, total_out


# ── Agent: Review Sub-Agent ───────────────────────────────────────────────────

def review_sub_agent(raw_answer: str, question: str) -> tuple:
    """
    Cleans the final answer to a plain number or percentage.

    Rules applied:
      * Remove currency symbols ($, pound, euro, etc.)
      * Remove commas from numbers
      * Keep % sign for percentages
      * Keep minus sign for negatives
      * "yes" / "no" pass through unchanged
      * No explanation — value only

    Returns
    -------
    clean_answer : str
    inp, out : token counts
    """
    system = """\
You are a financial answer formatter.

Return the answer as a clean value with NO explanation.

Formatting rules:
* Remove ALL currency symbols ($, pound sign, euro sign, etc.)
* Remove ALL commas from numbers  ->  "206,588"  becomes  "206588"
* Keep the % sign if the answer is a percentage  ->  "14.14%"
* Keep a leading minus for negatives  ->  "-25587"
* If the answer is "yes" or "no", return it unchanged
* Output ONLY the cleaned value — nothing else
"""
    user = f"""\
Question : {question}
Raw answer: {raw_answer}

Return the cleaned value.\
"""
    text, inp, out = _call(system, user, max_tokens=64)
    return text, inp, out


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def _run_conversation(doc: dict, turns: list) -> tuple:
    """
    Runs the full multi-agent pipeline for every conversation turn.
    Opens a single MCP connection shared across all program_sub_agent calls.

    Returns (all_responses, total_input_tokens, total_output_tokens).
    """
    server_params = StdioServerParameters(command=PYTHON, args=[CALCULATOR_SERVER])

    # Catch all exceptions inside the MCP context managers so they exit cleanly
    # before re-raising. Without this, anyio's internal TaskGroup wraps the
    # real exception in an ExceptionGroup, hiding the root cause.
    _exc = None
    _result = None

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            try:
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

                history: list = []
                all_responses: list = []
                total_inp = total_out = 0

                for turn in turns:
                    question = turn["user"]

                    # ── Planner Phase 1: classify ─────────────────────────────
                    plan, inp, out = planner_classify(question, history)
                    total_inp += inp; total_out += out

                    turn_type        = plan.get("turn_type", "Number")
                    is_type2         = plan.get("type2_question", False)
                    number_questions = plan.get("number_questions", [])

                    # ── type2_sub_agent (if needed) ───────────────────────────
                    if is_type2:
                        number_questions, inp, out = type2_sub_agent(question, history)
                        total_inp += inp; total_out += out

                    # ── document_sub_agent ────────────────────────────────────
                    retrieved_values, inp, out = document_sub_agent(number_questions, doc)
                    total_inp += inp; total_out += out

                    # ── route on turn_type ────────────────────────────────────
                    if turn_type == "Number":
                        # Single lookup — no arithmetic
                        raw_answer = next(
                            (v for v in retrieved_values.values() if v != "NOT_FOUND"),
                            "",
                        )

                    else:  # "Program"
                        # Planner Phase 2: build calculation plan
                        calc_plan, inp, out = planner_program(
                            question, retrieved_values, history
                        )
                        total_inp += inp; total_out += out

                        all_values  = calc_plan.get("all_values", retrieved_values)
                        calculation = calc_plan.get("calculation", "")

                        # program_sub_agent: execute via MCP tools
                        raw_answer, inp, out = await program_sub_agent(
                            all_values, calculation, session, tools
                        )
                        total_inp += inp; total_out += out

                    # ── review_sub_agent ──────────────────────────────────────
                    final_answer, inp, out = review_sub_agent(raw_answer, question)
                    total_inp += inp; total_out += out

                    history.append({"question": question, "answer": final_answer})
                    all_responses.append(final_answer)

                _result = (all_responses, total_inp, total_out)

            except Exception as e:
                _exc = e  # store so MCP context managers can exit cleanly

    if _exc is not None:
        raise _exc
    return _result


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
    all_responses, total_inp, total_out = loop.run_until_complete(
        _run_conversation(doc, turns)
    )

    return {
        "output": "\n\n---\n\n".join(all_responses),
        "tokenUsage": {
            "total":      total_inp + total_out,
            "prompt":     total_inp,
            "completion": total_out,
        },
    }
