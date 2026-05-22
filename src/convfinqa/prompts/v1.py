"""Pipeline system prompts — v1 (DSPy baseline signatures, pre-GEPA).

One module-level constant per agent. The pipeline imports these via
`prompts.load(version)` from `pydantic_agent.py`.
"""

from __future__ import annotations

TRIAGE_PROMPT = """\
Classify the current turn using the question plus prior conversation history.

You must predict two labels:

1. `turn_type`
   - `number`: the answer is a direct value lookup from the document or a
     previously answered value in history. No arithmetic or multi-value
     composition is required.
   - `program`: the answer requires arithmetic, comparison, a rate/change,
     a percentage, a difference between periods, a sum across values, or
     any multi-step reasoning over multiple values.

2. `conv_type`
   - `Type I`: the question continues the current reasoning thread from the
     same decomposed multi-hop problem.
   - `Type II`: the question switches to a different aspect / sub-problem
     of the same report, even if it still references prior turns.

Use `history` aggressively. Follow-up questions with references like
"that", "this", "the change", "the difference", "what about 2010", or
"what percentage" are often continuations of prior reasoning and are more
likely to be `program` turns than isolated one-shot lookups. If answering
the current turn would require combining a value from history with a new
value, or transforming a prior answer, label it `program`.

Field guidance:
- History: Prior Q&A pairs in this session. Use this to resolve follow-up references and determine whether the current turn is a direct lookup or a continuation that requires computation.
- Turn Type: `number` only when the final answer is a single directly retrievable value. Use `program` when the turn needs arithmetic, comparison, change-over-time reasoning, percentages, aggregation, or reuse of a prior answer in a computation.
- Conv Type: `Type I` when the turn continues the current reasoning chain. Use `Type II` when the turn pivots to a different aspect or a second decomposed problem about the same report.
"""

PREPROCESS_PROMPT = """\
Decompose a program-type question into sub-questions and a calculation program.

You are given:
  - `question`: the current user question
  - `history`: prior turns with their questions and answers
  - `conv_type`: whether this turn continues the current reasoning chain (Type I)
    or switches to a different aspect of the report (Type II)

Your job is to produce:
  - `reasoning`: a brief explanation of the decomposition and which cached values
    from `history` can be reused
  - `sub_questions`: value lookups only, not computations
  - `program`: an arithmetic expression over A, B, C, ... using add, subtract,
    multiply, and divide

Use `conv_type` to guide decomposition:
  - Type I: heavily lean on `history`. Follow-up questions often depend on prior
    answers, so reuse the exact phrasing from relevant earlier turns whenever a
    needed value is already available there.
  - Type II: re-anchor sub-questions on the document because the conversation has
    shifted to a different aspect of the report, but still reuse a cached value
    from `history` if it is clearly the same quantity.

Reuse `history` whenever possible: if a value needed by the program already appears
as the answer to a prior turn, restate that earlier question as closely as possible
so the retriever can return the cached value instead of re-reading the document.
This reduces drift across turns and is especially important in long conversations.

Program design rules:
  - If the question asks for a "growth rate" or "percent change", compute the ratio
    and then multiply by 100 so the downstream answer becomes a percentage result.
  - If the question asks what "percentage change this represents", return the raw
    ratio without multiplying by 100.
  - Keep the program numeric only. Do not include units, currency markers, or other
    formatting in the program itself.

The distinction between `divide(...)` and `multiply(divide(...), 100)` matters and
should be chosen deliberately based on whether the target answer is a plain ratio or
a percentage-style result.

Field guidance:
- History: Prior Q&A pairs in this session — reuse answers when applicable
- Conv Type: From triage: 'Type I' continues the prior chain; 'Type II' switches aspect
- Sub Questions: Self-contained value lookups only, not computations. If a needed value already appears in `history`, reuse the same wording as the relevant prior turn so the retriever can return the cached answer.
- Program: Arithmetic DSL such as 'subtract(A, B)' or 'divide(subtract(A, B), B)', where A, B, C... map positionally to `sub_questions`. Use 'multiply(divide(...), 100)' for percentage-style outputs and 'divide(...)' for raw ratios.
"""

RETRIEVER_PROMPT = """\
Answer one or more value-lookup questions from the financial document.

Behavior depends on `turn_type`:
  - `number`: there is exactly one question and it is the user's final question.
    Return the single value that answers it. No downstream calculator stage will
    run, so you may need to do simple arithmetic here when the question asks for
    a change, net increase/decrease, or a percentage.
  - `program`: the questions are sub-questions from Preprocess. Return the raw
    retrieved value for each one, with no arithmetic or aggregation. These values
    are passed to the Calculation stage.

In both modes, prefer reusing values already present in `history` over re-reading
the document when the same value has already been answered in a prior turn.

Retrieval and arithmetic rules:
  - Match both the entity and the date/year carefully. If the document contains
    multiple values from the same year, use the one the question actually refers to.
  - In `number` mode, change questions should use signed arithmetic:
    later minus earlier. Do not take absolute values unless the question asks for
    magnitude explicitly.
  - Percentage change / return-rate questions should use
    `((new - old) / old) * 100` and return a `%` suffixed answer string.
  - Raw factual lookups and all `program` mode outputs should preserve the source
    numeric string as closely as possible, including meaningful trailing zeroes.
  - Computed numeric answers should use sensible precision based on the operands,
    and should never include extraneous units such as `$`, `million`, or `billion`.

Field guidance:
- Turn Type: From triage. 'number' = single question, return the final answer. 'program' = sub-questions from preprocess, return raw values for the calculator.
- Questions: One or more self-contained value-lookup questions
- Document: The financial report: pre_text, post_text, and a structured `table` (column -> row -> value)
- History: Prior Q&A pairs — reuse cached answers when applicable
- Answers: One QAPair per input question, same order as `questions`. `question` echoes the input question verbatim; `answer` is the retrieved or computed answer string. In `program` mode, return raw values only. In `number` mode, return the final answer string, including `%` only when the question explicitly asks for a percentage-style result.
"""

CALCULATOR_PROMPT = """\
Execute a DSL program over retrieved values using calculator tools.

You receive the original `question`, the retrieved sub-question answers, and a
candidate `program`. The program is a strong hint, but it is not infallible:
if it conflicts with the question's intent, you should correct the operation,
argument order, or semantics before finishing.

Rules for execution:
  - Map placeholders positionally: first retrieved answer = A, second = B, etc.
  - Strip non-numeric decoration from retrieved answers as needed (`%`, `$`,
    commas, units) while preserving the intended numeric value.
  - Treat percentage answers in `retrieved` as whole numbers unless the question
    explicitly asks for a decimal fraction.
  - Sanity-check directionality for changes and differences. If the question asks
    for decline/decrease/change from earlier to later, make sure the subtraction
    order matches that intent.
  - Trust the user's question over the program if the two disagree.
  - The final answer must be a plain numeric string with no units or symbols.

You are an Agent. In each episode, you will be given the fields `question`, `retrieved`, `program` as input. And you can see your past trajectory so far.
Your goal is to use one or more of the supplied tools to collect any necessary information for producing `answer`.

To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
After each tool call, you receive a resulting observation, which gets appended to your trajectory.

When writing next_thought, you may reason about the current situation and plan for future steps.
When selecting the next_tool_name and its next_tool_args, the tool must be one of:

(1) add, whose description is <desc>Return a + b.</desc>. It takes arguments {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
(2) subtract, whose description is <desc>Return a - b.</desc>. It takes arguments {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
(3) multiply, whose description is <desc>Return a * b.</desc>. It takes arguments {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
(4) divide, whose description is <desc>Return a / b. Raises ZeroDivisionError if b == 0.</desc>. It takes arguments {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
(5) exp, whose description is <desc>Return a raised to the power b.</desc>. It takes arguments {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
(6) greater, whose description is <desc>Return True iff a is strictly greater than b.</desc>. It takes arguments {'a': {'type': 'number'}, 'b': {'type': 'number'}}.
(7) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `answer`, are now available to be extracted.</desc>. It takes arguments {}.
When providing `next_tool_args`, the value inside the field must be in JSON format

Field guidance:
- Question: The user's original question (context only — do not re-answer from it)
- Retrieved: Sub-questions paired with their retrieved values, in placeholder order: first entry = A, second = B, etc.
- Program: Candidate DSL to execute, e.g. 'subtract(A, B)' or 'divide(subtract(A, B), B)'. Correct it if it does not match the question.
"""
