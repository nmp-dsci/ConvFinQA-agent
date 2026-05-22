name: "ConvFinQA Pydantic AI Port — Mirror Optimized DSPy Pipeline"
description: |

## Purpose
Port the optimized DSPy multi-agent pipeline in `dspy_agent.py` to a Pydantic AI implementation in `pydantic_agent.py`, using the GEPA-optimized system prompts persisted in `runs/gepa_smoke_20260429_204159/optimized_runner.json` as the per-stage system prompts. Same 4-stage architecture, same calculator tools, same evaluation harness — just a different agent framework.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create `pydantic_agent.py` — a Pydantic AI implementation of the same 4-stage ConvFinQA pipeline as `dspy_agent.py`:

1. **Triage** agent (single-shot classifier → `turn_type`, `conv_type`)
2. **Preprocess** agent (program-only; emits `sub_questions` + DSL `program`)
3. **Retriever** agent (looks up values in document, reuses history)
4. **Calculator** agent (tool-using agent that executes DSL via 6 calc tools)

Each agent's `system_prompt` is the corresponding `signature.instructions` string loaded from `runs/gepa_smoke_20260429_204159/optimized_runner.json`. Field-level descriptions from the optimized signatures are appended to the instructions so the per-input/output guidance survives the port.

## Why
- **Reuse the optimization**: GEPA spent compute discovering better prompts for triage/preprocess/retrieve/calculate. Those prompts live in `optimized_runner.json` and are framework-agnostic — they're just text. Loading them into Pydantic AI gets us the optimization without re-running GEPA.
- **Pydantic AI is the production-target framework**: structured outputs via Pydantic models, first-class tool calling, Logfire integration, and easier deployment than a DSPy module that pulls in MLflow.
- **Apples-to-apples comparison**: with the same prompts, same models, same test set, and same `analyze_predictions` slicing, we can directly measure whether anything regresses in the port (it shouldn't — but if it does we'll see exactly which slice).
- **Future iteration**: once ported, future GEPA-optimized prompts can be loaded the same way (one JSON path swap), so the optimization loop and the production runtime stay decoupled.

## What
A single-file `pydantic_agent.py` that:

- Loads optimized instructions from `runs/<gepa_name>/optimized_runner.json` at startup (path overridable via env var)
- Defines five Pydantic models for the per-stage I/O (`TriageOut`, `PreprocessOut`, `RetrievedValue`, `RetrievedValues`, `CalcOut`)
- Builds four `pydantic_ai.Agent` instances with `output_type` set to those models, `system_prompt` set to the loaded instructions
- Wraps them in a `ConversationRunner` class that walks all turns of one conversation in order, threading conversation history exactly like the DSPy version
- Reuses the existing `qa_data` / `conv_examples_test` / `analyze_predictions` plumbing from `dspy_agent.py` (imports, doesn't duplicate)
- Writes per-turn predictions to `runs/<gepa_name>/pydantic_predictions.csv` and prints the same accuracy slices

### Success Criteria
- [ ] `uv run python pydantic_agent.py` runs end-to-end on the 100-conversation test set without unhandled exceptions
- [ ] **Same records as DSPy**: `pydantic_agent.py` evaluates on the *exact same* `(report_id, turn_index)` rows as `dspy_agent.py` — verified by importing `conv_examples_test` from `dspy_agent` (not rebuilding it) and by a post-hoc `pd.merge(..., how="outer", indicator=True)` showing no `left_only`/`right_only` rows
- [ ] Overall turn accuracy within ±5 pp of `dspy_agent.py` loaded from the same `optimized_runner.json` (sanity check that the port preserves behavior)
- [ ] Per-slice deltas (turn_type, conv_type, q_order) within ±10 pp; any slice exceeding this is flagged for investigation, not silently accepted
- [ ] All 4 agents use the *exact* `signature.instructions` strings from `optimized_runner.json` (verifiable: hash the loaded prompt vs. the file's value)
- [ ] **I/O parity with DSPy signatures**: TriageOut / PreprocessOut / RetrievedValues each include `reasoning: str` (matching `dspy.ChainOfThought`'s auto-added field); CalcOut does not (matching `dspy.ReAct`'s output shape). Field names and types match the table in "Signature Parity" exactly.
- [ ] **Wire-format parity**: every sub-agent's user message is rendered via `_render_chat_inputs(...)` and contains `[[ ## field_name ## ]]` markers using the exact DSPy input field names (`question`, `history`, `conv_type`, `turn_type`, `questions`, `document`, `retrieved`, `program`)
- [ ] Calculator agent has all 6 tools registered (`add`, `subtract`, `multiply`, `divide`, `exp`, `greater`)
- [ ] `pytest tests/test_pydantic_agent.py -v` passes
- [ ] `ruff check pydantic_agent.py --fix && ruff format pydantic_agent.py` — zero errors
- [ ] `predictions.csv` and `predictions_joined.csv` written under `runs/<gepa_name>/` with `pydantic_` prefix
- [ ] Slice analysis runs and prints `turn_type`, `conv_type`, `q_order` cuts plus 2-way pivots (using the existing `analyze_predictions` helper)
- [ ] A `parity_report.csv` is written under `runs/<gepa_name>/` — one row per (report_id, turn_index) with both predictions side-by-side and an `agree` flag, so disagreements are inspectable

## Test Set Parity — Same Records as DSPy

Parity is a hard requirement: the Pydantic AI port must be evaluated on the *exact same conversations and turns* as `dspy_agent.py`, otherwise any accuracy comparison is meaningless.

**Mechanism (single source of truth):**
- `dspy_agent.py` constructs `conv_examples_test` deterministically from `data/convfinqa_dataset.json` using fixed seeds (`random_state=42`, `random.Random(42)`). The current build is: original 100 sampled report_ids → 60 train / 40 test (seeded shuffle), plus an additional 60 disjoint report_ids appended to test → **100 test conversations**.
- `pydantic_agent.py` MUST `from dspy_agent import conv_examples_test` and iterate over that list directly. Do **not** re-derive a test set from `qa_data` independently — even with the same seeds, divergence is one off-by-one away.
- `(report_id, turn_index)` is the join key between the two prediction CSVs. `turn_index` is the 0-based position within `q_order`-sorted turns for a report. `dspy_agent.analyze_predictions` already uses this convention; `pydantic_agent.py` must too.

**Verification at runtime:**
- After `pydantic_predictions.csv` is written, build a parity report:
  ```python
  import pandas as pd
  d = pd.read_csv("runs/<GEPA_NAME>/predictions.csv")
  p = pd.read_csv("runs/<GEPA_NAME>/pydantic_predictions.csv")
  m = d.merge(p, on=["report_id", "turn_index", "question", "gold_answer"],
              how="outer", suffixes=("_dspy", "_pyd"), indicator=True)
  assert (m["_merge"] == "both").all(), \
      f"Test-set drift: {(m['_merge'] != 'both').sum()} rows not in both"
  m["agree"] = m["correct_dspy"] == m["correct_pyd"]
  m.to_csv("runs/<GEPA_NAME>/parity_report.csv", index=False)
  ```
  This both *asserts* identical record coverage and produces a CSV showing every disagreement for inspection.

**Comparison reporting (printed at end of `pydantic_agent.py` run):**
- Overall: `dspy_acc`, `pyd_acc`, `delta_pp`, `agree_rate`
- Per slice (turn_type, conv_type, q_order): `dspy_acc`, `pyd_acc`, `delta_pp`, `n`
- Counts of `only_dspy_correct`, `only_pyd_correct`, `both_correct`, `both_wrong`

This goes in `pydantic_agent.py` itself (a `compare_runs(dspy_csv, pyd_csv)` function called from `__main__` when both files exist), so the parity check runs every time, not just in CI.

## Signature Parity — DSPy → Pydantic AI Field-by-Field

The Pydantic AI agents must accept and emit the *same* fields, with the *same names and types*, as the DSPy signatures the optimized prompts were tuned against. Any rename or drop changes what the LLM sees and breaks the optimization.

| DSPy module | Predictor type | Inputs (name : type) | Outputs (name : type) |
|---|---|---|---|
| `TriageSignature` | `ChainOfThought` | `question: str` | `reasoning: str` (auto from ChainOfThought), `turn_type: Literal["number","program"]`, `conv_type: Literal["Type I","Type II"]` |
| `PreprocessSignature` | `ChainOfThought` | `question: str`, `history: str`, `conv_type: Literal["Type I","Type II"]` | `reasoning: str`, `sub_questions: list[str]`, `program: str` |
| `RetrieverSignature` | `ChainOfThought` | `turn_type: Literal["number","program"]`, `questions: list[str]`, `document: Document`, `history: str` | `reasoning: str`, `answers: list[QAPair]` |
| `CalculationSignature` | `ReAct(tools=CALCULATOR_TOOLS, max_iters=8)` | `question: str`, `retrieved: list[QAPair]`, `program: str` | `answer: str` (ReAct's `trajectory` is internal — not part of the signature output) |

### Implications for the Pydantic AI port

1. **Add `reasoning: str` to TriageOut, PreprocessOut, and RetrievedValues.** ChainOfThought's reasoning isn't decorative — the prompts were optimized assuming the model writes it before the structured fields. Dropping it changes elicitation behavior.
2. **Use exact DSPy field names in the wire format**, not paraphrases. `questions` not `lookup_questions`, `retrieved` not `values`, `document` not `doc`.
3. **Render inputs in DSPy `ChatAdapter` format** — `[[ ## field_name ## ]]\n{value}` blocks — because the optimized instructions reference fields by these markers in places. Provide a `_render_chat_inputs(fields: dict[str, Any]) -> str` helper that produces this exact format. The user message becomes a sequence of `[[ ## name ## ]]\n{value}\n` blocks for each input field, in the order the DSPy signature declared them.
4. **Calculator uses tools, not `reasoning`.** `dspy.ReAct` does not auto-add a `reasoning` field — the trajectory IS the reasoning, captured in tool-call history (which Pydantic AI's `RunResult.all_messages()` already exposes). `CalcOut` correctly stays `answer: str` only.
5. **Document goes in via `document: Document` field name** with the value rendered as `document.model_dump_json(indent=2)`. DSPy passed the typed `Document` and let ChatAdapter serialize it; we serialize manually but keep the field name identical.
6. **`retrieved` for the calculator** must be the JSON list of `QAPair` dicts in the same order as `program` placeholders (A, B, ...). DSPy passes `list[QAPair]` directly; we render `[qa.model_dump() for qa in r.answers]`.

## All Needed Context

### Documentation & References
```yaml
# MUST READ — Include these in your context window
- file: dspy_agent.py
  why: |
    The reference implementation. Key sections to mirror:
      - Lines ~125-220: Pydantic models (QAPair, HistoryTurn, ConversationHistory,
        Document, AgentResponse) — REUSE these by importing.
      - Lines ~220-235: Calculator tools (add/subtract/multiply/divide/exp/greater)
        — REUSE by importing CALCULATOR_TOOLS.
      - Lines ~520-595: ConversationRunner.forward / _run_turn — this is the
        per-turn flow to mirror in Pydantic AI.
      - Lines ~660-695: analyze_predictions — REUSE by import.
      - Lines ~485-495: conv_examples_test construction — REUSE by import.

- file: runs/gepa_smoke_20260429_204159/optimized_runner.json
  why: |
    Source of system prompts. Structure:
      {
        "triage.predict":              {"signature": {"instructions": "...", "fields": {...}}, ...},
        "preprocess.predict":          {"signature": {"instructions": "...", "fields": {...}}, ...},
        "retriever.predict":           {"signature": {"instructions": "...", "fields": {...}}, ...},
        "calculator.react":            {"signature": {"instructions": "...", "fields": {...}}, ...},
        "calculator.extract.predict":  {"signature": {"instructions": "...", "fields": {...}}, ...},
        "metadata": {...}
      }
    The four prompts to load are: triage.predict, preprocess.predict,
    retriever.predict, calculator.react.
    `calculator.extract.predict` is DSPy's internal ReAct extractor — Pydantic AI
    handles tool-call extraction natively, so we IGNORE this one (notes below).
    Instruction lengths confirm they were optimized:
      triage=67 (~baseline), preprocess=2377, retriever=5317, calculator=5044.

- file: CLAUDE.md
  why: |
    AI Agent Architecture section explains the 4-stage flow conceptually.
    Models / Caching section covers DeepSeek + DSPY_CACHEDIR — the LM cache is
    DSPy-specific and does NOT apply to Pydantic AI; pydantic_agent.py uses the
    native LM SDK cache (or none).

- file: pyproject.toml
  why: |
    Confirms deps already pinned: pydantic-ai>=0.2, logfire[httpx,pydantic-ai],
    tenacity>=8.0.0, anthropic>=0.83.0. NO new deps required.

- url: https://ai.pydantic.dev/agents/
  why: Agent creation, system_prompt, output_type, deps_type, tool registration.

- url: https://ai.pydantic.dev/api/agent/
  why: Agent.run() returns RunResult with .output (typed) and .usage().

- url: https://ai.pydantic.dev/multi-agent-applications/
  why: |
    NOT using agent-as-tool here — the 4 stages are orchestrated by plain Python
    (a class method), same shape as dspy_agent.py's ConversationRunner.

- url: https://ai.pydantic.dev/models/
  why: |
    DeepSeek isn't a first-party Pydantic AI provider. Two options:
      (a) Use OpenAIModel with custom base_url='https://api.deepseek.com'
          and model='deepseek-chat' — DeepSeek is OpenAI-compatible.
      (b) Use Anthropic Claude (default per CLAUDE.md project conventions)
          via 'anthropic:claude-haiku-4-5' or 'anthropic:claude-sonnet-4-6'.
    DEFAULT TO (a) so model parity with dspy_agent.py is preserved — that's
    what the GEPA prompts were optimized against. Allow override via env var.

- url: https://logfire.pydantic.dev/docs/integrations/pydantic-ai/
  why: One-line instrumentation `logfire.instrument_pydantic_ai()`.
```

### Current Codebase tree (relevant)
```bash
.
├── dspy_agent.py                              # Reference implementation
├── data.py                                    # qa_data loader
├── data/convfinqa_dataset.json
├── runs/
│   └── gepa_smoke_20260429_204159/
│       ├── optimized_runner.json              # SOURCE of system prompts
│       └── ...
├── tests/
│   ├── __init__.py
│   └── test_agent.py
└── pyproject.toml                             # All deps already present
```

### Desired Codebase tree
```bash
.
├── pydantic_agent.py                          # NEW — main implementation
├── tests/
│   └── test_pydantic_agent.py                 # NEW — test suite
└── runs/gepa_smoke_20260429_204159/
    ├── pydantic_predictions.csv               # NEW (written by evaluate())
    ├── pydantic_predictions_joined.csv        # NEW (written by analyze_predictions)
    └── parity_report.csv                      # NEW (written by compare_runs)
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: optimized_runner.json contains DSPy's ChatAdapter format markers
#   (e.g. "[[ ## field_name ## ]]") in field descriptions. The `instructions`
#   string itself is plain prose and reusable as-is. The `fields` dict carries
#   per-field "prefix"/"description" which DSPy uses for output parsing. We
#   DO NOT need the prefix machinery — Pydantic AI handles structured output
#   via output_type. We DO want the field descriptions to give the LLM context
#   about each output field, so append them to the system prompt as a
#   "Field guidance" section.

# CRITICAL: The DSPy retriever signature has `turn_type` as an INPUT — it
#   branches behavior between number-mode (return final answer) and program-mode
#   (return raw retrieved values for the calculator). In Pydantic AI we replicate
#   this by passing turn_type into the user-message string template, NOT by
#   making it a separate Pydantic field on the agent's input.

# CRITICAL: DSPy's `dspy.ReAct(CalculationSignature, tools=CALCULATOR_TOOLS,
#   max_iters=8)` is a tool-using loop. Pydantic AI's Agent does this natively
#   when tools are registered. The `calculator.react` instructions in
#   optimized_runner.json are written for ReAct (mention thought/action/
#   observation cycles in some places). They're still useful as system prompt —
#   Pydantic AI's agent will use the calc tools to satisfy them, just without
#   the explicit ReAct scaffolding text. Do NOT try to mimic ReAct's
#   trajectory format manually.

# CRITICAL: `calculator.extract.predict` exists in optimized_runner.json
#   because DSPy's ReAct uses a separate Predict step to extract the final
#   answer from the trajectory. Pydantic AI extracts the typed output directly
#   from the LLM response — IGNORE this entry. Do NOT load it as a system prompt.

# CRITICAL: I/O parity is non-negotiable. Each Pydantic AI agent's output_type
#   must declare the SAME fields, with the SAME names and types, as the DSPy
#   signature it replaces. Specifically: ChainOfThought-derived predictors
#   (triage, preprocess, retriever) auto-add a `reasoning: str` output that
#   the optimized prompts elicit — TriageOut/PreprocessOut/RetrievedValues
#   MUST include `reasoning: str`. CalcOut does NOT (ReAct's trajectory is
#   the reasoning, not a structured output field).

# CRITICAL: Inputs must be rendered in DSPy ChatAdapter format —
#   `[[ ## field_name ## ]]\n{value}\n` blocks, in the same field order the
#   DSPy signature declared. The optimized instructions sometimes reference
#   fields by these markers; using prose labels ("Question:", "Doc:") is a
#   silent format mismatch that degrades accuracy. Use _render_chat_inputs()
#   with EXACT field names: question, history, conv_type, turn_type,
#   questions, document, retrieved, program.

# CRITICAL: ConversationHistory in dspy_agent.py serializes prior turns to a
#   text block via `as_text()`. Reuse the same text format so the prompts
#   (which were OPTIMIZED against that exact format) interpret history the
#   same way. Do NOT switch to Pydantic AI's `message_history` parameter for
#   this — it would change the format.

# CRITICAL: The Document model serializes table as a dict-of-dicts. The
#   retriever instructions reference this layout in places. Pass the document
#   to the retriever agent via `document.model_dump_json(indent=2)` in the
#   user message — same shape DSPy uses internally.

# GOTCHA: Pydantic AI is async-first. Use `await agent.run(...)`. Wrap the
#   conversation walk in `asyncio.run(...)` from __main__. Reuse the per-conv
#   parallelism via `asyncio.gather` over conversations (NOT over turns within
#   a conversation — turns are sequential because of the history dependency).

# GOTCHA: DeepSeek via OpenAI-compatible API: model name is "deepseek-chat" or
#   "deepseek-reasoner", base_url="https://api.deepseek.com/v1". The DeepSeek
#   API key env var is DEEPSEEK_API_KEY (already used by dspy_agent.py).

# GOTCHA: Logfire requires LOGFIRE_TOKEN to actually ship traces. Without it,
#   `logfire.configure()` is a no-op locally. Don't make it a hard dependency.

# GOTCHA: `optimized_runner.json` stores `signature.fields` as a LIST of
#   {prefix, description} dicts, NOT a dict keyed by field name. Treating it
#   like a dict (`.items()`) crashes. The list also has no field-name key, so
#   you can't reliably attribute descriptions to fields. The right move is
#   to use `signature.instructions` only — that's where GEPA's optimization
#   actually lives. Don't try to reconstruct field guidance from this list.

# GOTCHA: Importing dspy_agent at module level runs ALL of its top-level code:
#   loading data/convfinqa_dataset.json, building qa_data, configuring
#   dspy.LM(deepseek/...). That LM construction reads DEEPSEEK_API_KEY at
#   import time. So `load_dotenv(...)` MUST run BEFORE
#   `from dspy_agent import ...` or you'll get a KeyError before any of your
#   own code runs. Also: import takes a few seconds — fine for batch eval,
#   bad for short-lived CLI invocations.

# GOTCHA: The optimized prompts may be long enough that prompt caching matters
#   for cost. Pydantic AI doesn't auto-enable prompt caching for Anthropic —
#   if you switch to Anthropic, set cache_control on the system prompt block.
#   For DeepSeek (default), DeepSeek's API does context caching automatically.
```

## Implementation Blueprint

### Module Skeleton (Assembly Order)

`pydantic_agent.py` should be assembled top-to-bottom in this order — each block depends only on the ones above it:

```python
"""Pydantic AI port of the optimized DSPy ConvFinQA pipeline."""
# 1. Std-lib imports
from __future__ import annotations
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any, Literal

# 2. Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# 3. Env loading — must run BEFORE importing dspy_agent
#    (dspy_agent.py reads DEEPSEEK_API_KEY at module-import time
#    when constructing dspy.LM, and would crash if it's unset).
load_dotenv(Path.home() / ".env")

# 4. Reused symbols from dspy_agent — importing it executes the module
#    (loads convfinqa_dataset.json, qa_data, configures DSPy LM, builds
#    conv_examples_test, defines _DOCS). This is intentional: that
#    construction IS the source of truth for the test set.
from dspy_agent import (              # noqa: E402  (order matters — see above)
    Document, QAPair, ConversationHistory,
    CALCULATOR_TOOLS, _DOCS,
    qa_data, conv_examples_test,
    numeric_match, analyze_predictions,
)

# 5. Constants
GEPA_NAME = os.environ.get("GEPA_NAME", "gepa_smoke_20260429_204159")
PROMPTS_PATH = Path("runs") / GEPA_NAME / "optimized_runner.json"
TurnType = Literal["number", "program"]
ConvType = Literal["Type I", "Type II"]

# 6. Prompt loader + PROMPTS dict (see "Loading the Optimized Prompts")
# 7. Output models (see "Per-Stage Pydantic Models")
# 8. LM + the four agents + tool registration (see "The Four Agents")
# 9. _render_chat_inputs + ConversationRunner (see "Conversation Runner")
# 10. evaluate() + compare_runs() (see "Evaluation Harness")
# 11. if __name__ == "__main__": main entry (see "Evaluation Harness")
```

### Loading the Optimized Prompts

```python
# ── prompt loading ──────────────────────────────────────────────────────────
def _load_optimized_prompts(path: Path) -> dict[str, str]:
    """Load per-stage system prompts from a DSPy optimized_runner.json.

    Returns {triage, preprocess, retriever, calculator} mapping to the
    `signature.instructions` string for each stage. The optimized instructions
    are what GEPA grew (preprocess=2.4kB, retriever=5.3kB, calculator=5.0kB) —
    everything learned by the optimization is in there.

    The signature `fields` list is intentionally NOT appended:
      - Input field descriptions are DSPy ChatAdapter placeholder templates
        (e.g. description="${question}") — adding them is noise.
      - Output field descriptions duplicate text already in `instructions`
        after GEPA's reflection passes.
      - The list-of-{prefix,description} dicts has no field-name key, so we
        couldn't reliably attribute descriptions even if we wanted to.

    `calculator.extract.predict` is intentionally skipped — Pydantic AI handles
    typed-output extraction natively (see CRITICAL gotcha).
    """
    raw = json.loads(path.read_text())
    mapping = {
        "triage":     "triage.predict",
        "preprocess": "preprocess.predict",
        "retriever":  "retriever.predict",
        "calculator": "calculator.react",
    }
    return {
        short: raw[key]["signature"]["instructions"].rstrip()
        for short, key in mapping.items()
    }

PROMPTS = _load_optimized_prompts(PROMPTS_PATH)
```

### Per-Stage Pydantic Models

(Imports already covered in the Module Skeleton above.)

```python
# ── output models — mirror DSPy signatures ──────────────────────────────────
class TriageOut(BaseModel):
    """Output of the triage agent — mirrors DSPy TriageSignature outputs.
    `reasoning` is included to match dspy.ChainOfThought's auto-added field."""
    reasoning: str
    turn_type: TurnType
    conv_type: ConvType


class PreprocessOut(BaseModel):
    """Output of the preprocess agent (program-mode only) — mirrors PreprocessSignature."""
    reasoning: str
    sub_questions: list[str]
    program: str  # DSL like "divide(subtract(A, B), B)"


class RetrievedValues(BaseModel):
    """Output of the retriever agent — mirrors RetrieverSignature.
    `answers[i].question` MUST echo `questions[i]` verbatim; `answers[i].answer`
    is the raw retrieved value (no computation, no aggregation)."""
    reasoning: str
    answers: list[QAPair]


class CalcOut(BaseModel):
    """Output of the calculator agent — mirrors CalculationSignature.
    No `reasoning` field: dspy.ReAct does not auto-add reasoning; the tool-call
    trajectory IS the reasoning trace, captured by Pydantic AI in
    RunResult.all_messages()."""
    answer: str  # final numeric answer as string
```

### The Four Agents

```python
# ── agents ──────────────────────────────────────────────────────────────────
# DeepSeek via OpenAI-compatible endpoint — matches dspy_agent.py.
# Fail fast if the key is missing; do NOT silently fall back to another model.
if "DEEPSEEK_API_KEY" not in os.environ:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. The optimized prompts were tuned against "
        "DeepSeek; using a different model breaks the optimization premise."
    )
_deepseek_provider = OpenAIProvider(
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)
LM_MINI = OpenAIModel("deepseek-chat", provider=_deepseek_provider)

triage_agent = Agent(
    LM_MINI,
    output_type=TriageOut,
    system_prompt=PROMPTS["triage"],
)

preprocess_agent = Agent(
    LM_MINI,
    output_type=PreprocessOut,
    system_prompt=PROMPTS["preprocess"],
)

retriever_agent = Agent(
    LM_MINI,
    output_type=RetrievedValues,
    system_prompt=PROMPTS["retriever"],
)

calculator_agent = Agent(
    LM_MINI,
    output_type=CalcOut,
    system_prompt=PROMPTS["calculator"],
)

# Register the 6 calc tools on the calculator agent.
# CALCULATOR_TOOLS is imported from dspy_agent.py — they're plain Python fns
# that Pydantic AI accepts directly via Agent.tool_plain.
for fn in CALCULATOR_TOOLS:
    calculator_agent.tool_plain(fn)
```

### Conversation Runner

```python
# ── input rendering — mirror DSPy ChatAdapter wire format ───────────────────
def _render_chat_inputs(fields: dict[str, Any]) -> str:
    """Render an ordered dict of input fields in DSPy ChatAdapter format.

    Produces:
        [[ ## name1 ## ]]
        {value1}
        [[ ## name2 ## ]]
        {value2}

    The optimized prompts in optimized_runner.json were tuned against this
    exact wire format. Field ORDER must match the DSPy signature's declared
    input order (Python dicts are insertion-ordered, so the caller controls
    this by the order they pass kwargs).

    Values that aren't strings are rendered with model_dump_json (Pydantic
    models), JSON dumps (lists/dicts), or str() (everything else).
    """
    import json as _json
    parts = []
    for name, value in fields.items():
        if isinstance(value, BaseModel):
            rendered = value.model_dump_json(indent=2)
        elif isinstance(value, (list, dict)):
            try:
                rendered = _json.dumps(
                    [v.model_dump() if isinstance(v, BaseModel) else v for v in value]
                    if isinstance(value, list) else value,
                    indent=2, default=str,
                )
            except TypeError:
                rendered = str(value)
        else:
            rendered = str(value)
        parts.append(f"[[ ## {name} ## ]]\n{rendered}")
    return "\n".join(parts)


# ── orchestrator (per-conversation, sequential turns) ───────────────────────
class ConversationRunner:
    """Walk all turns of one conversation, threading history.

    Mirrors dspy_agent.py:ConversationRunner. NOT a Pydantic AI agent itself —
    plain Python orchestration over the four typed agents. Each sub-agent
    receives inputs in DSPy ChatAdapter wire format with the EXACT field names
    the original signatures declared.
    """

    async def _run_turn(
        self,
        question: str,
        report_id: str,
        document: Document,
        conversation: ConversationHistory,
    ) -> str:
        hist_text = conversation.as_text()

        # TriageSignature inputs: question
        triage_msg = _render_chat_inputs({"question": question})
        triage = (await triage_agent.run(triage_msg)).output

        if triage.turn_type == "number":
            # RetrieverSignature inputs: turn_type, questions, document, history
            retr_msg = _render_chat_inputs({
                "turn_type": "number",
                "questions": [question],
                "document": document,
                "history": hist_text,
            })
            r = (await retriever_agent.run(retr_msg)).output
            answer = str(r.answers[0].answer)
            conversation.append(question=question, answer=answer, report_id=report_id)
            return answer

        # program path
        # PreprocessSignature inputs: question, history, conv_type
        pp_msg = _render_chat_inputs({
            "question": question,
            "history": hist_text,
            "conv_type": triage.conv_type,
        })
        pp = (await preprocess_agent.run(pp_msg)).output

        # RetrieverSignature inputs: turn_type, questions, document, history
        retr_msg = _render_chat_inputs({
            "turn_type": "program",
            "questions": list(pp.sub_questions),
            "document": document,
            "history": hist_text,
        })
        r = (await retriever_agent.run(retr_msg)).output

        # CalculationSignature inputs: question, retrieved, program
        calc_msg = _render_chat_inputs({
            "question": question,
            "retrieved": [qa.model_dump() for qa in r.answers],
            "program": pp.program,
        })
        calc = (await calculator_agent.run(calc_msg)).output
        answer = str(calc.answer)
        conversation.append(question=question, answer=answer, report_id=report_id)
        return answer

    async def run_conversation(
        self, report_id: str, questions: list[str]
    ) -> list[str]:
        document = _DOCS[report_id]  # imported from dspy_agent.py
        conversation = ConversationHistory()
        return [
            await self._run_turn(q, report_id, document, conversation)
            for q in questions
        ]
```

### Evaluation Harness

```python
# ── eval (mirrors dspy_agent.py main + analyze_predictions) ─────────────────
async def evaluate(report_ids_examples: list, max_concurrency: int = 8) -> Path:
    """Run the runner over every conversation, write predictions.csv."""
    sem = asyncio.Semaphore(max_concurrency)
    runner = ConversationRunner()

    async def _one(ex):
        async with sem:
            try:
                preds = await runner.run_conversation(ex.report_id, ex.questions)
            except Exception as e:
                print(f"  [error] {ex.report_id}: {e!r}")
                preds = []
            return ex, preds

    results = await asyncio.gather(*[_one(ex) for ex in report_ids_examples])

    out_dir = Path("runs") / GEPA_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pydantic_predictions.csv"

    import csv
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["report_id", "turn_index", "question",
                    "gold_answer", "pred_answer", "correct"])
        for ex, preds in results:
            for i, (q, g) in enumerate(zip(ex.questions, ex.gold_answers)):
                p = preds[i] if i < len(preds) else None
                w.writerow([
                    ex.report_id, i, q, g, p,
                    numeric_match(p, g) if p is not None else False,
                ])

    overall = sum(numeric_match(p, g)
                  for ex, preds in results
                  for p, g in zip(preds, ex.gold_answers)) / sum(
                      len(ex.gold_answers) for ex, _ in results)
    print(f"\nOverall turn accuracy: {overall:.1%}")
    return out_path


def compare_runs(dspy_csv: Path, pyd_csv: Path) -> Path:
    """Side-by-side parity report against the DSPy run on the same artifact.

    Asserts identical (report_id, turn_index, question, gold_answer) coverage
    so any test-set drift fails loudly. Writes parity_report.csv and prints
    overall + per-slice deltas plus agreement counts.
    """
    import pandas as pd
    d = pd.read_csv(dspy_csv).rename(columns={
        "pred_answer": "pred_dspy", "correct": "correct_dspy",
    })
    p = pd.read_csv(pyd_csv).rename(columns={
        "pred_answer": "pred_pyd", "correct": "correct_pyd",
    })
    m = d.merge(
        p, on=["report_id", "turn_index", "question", "gold_answer"],
        how="outer", indicator=True,
    )
    drift = (m["_merge"] != "both").sum()
    if drift:
        raise RuntimeError(
            f"Test-set drift: {drift} rows are not in both runs. "
            "pydantic_agent.py and dspy_agent.py must evaluate the same records."
        )
    m["agree"] = m["correct_dspy"] == m["correct_pyd"]

    # join in slice columns from qa_data so per-slice deltas mean the same
    # thing as analyze_predictions
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    qa["conv_type"] = qa["qa_split"].map({True: "Type II", False: "Type I"})
    m = m.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "conv_type"]],
        on=["report_id", "turn_index"], how="left",
    )

    out_path = pyd_csv.with_name("parity_report.csv")
    m.to_csv(out_path, index=False)

    print(f"\n=== Parity vs DSPy ({dspy_csv.name} ↔ {pyd_csv.name}) ===")
    print(f"n turns: {len(m)}")
    print(f"dspy acc: {m['correct_dspy'].mean():.1%}")
    print(f"pyd  acc: {m['correct_pyd'].mean():.1%}")
    print(f"delta:    {(m['correct_pyd'].mean() - m['correct_dspy'].mean())*100:+.1f} pp")
    print(f"agreement: {m['agree'].mean():.1%}")
    print(f"  both correct:    {((m.correct_dspy)  & (m.correct_pyd)).sum()}")
    print(f"  both wrong:      {((~m.correct_dspy) & (~m.correct_pyd)).sum()}")
    print(f"  only dspy right: {((m.correct_dspy)  & (~m.correct_pyd)).sum()}")
    print(f"  only pyd right:  {((~m.correct_dspy) & (m.correct_pyd)).sum()}")

    for col in ("turn_type", "conv_type", "q_order"):
        cut = m.groupby(col).agg(
            dspy_acc=("correct_dspy", "mean"),
            pyd_acc=("correct_pyd", "mean"),
            n=("correct_pyd", "size"),
        )
        cut["delta_pp"] = (cut["pyd_acc"] - cut["dspy_acc"]) * 100
        cut["dspy_acc"] = cut["dspy_acc"].map(lambda v: f"{v:.1%}")
        cut["pyd_acc"]  = cut["pyd_acc"].map(lambda v: f"{v:.1%}")
        cut["delta_pp"] = cut["delta_pp"].map(lambda v: f"{v:+.1f}")
        print(f"\nBy {col}:")
        print(cut.to_string())

    print(f"\nWrote {out_path}")
    return out_path


if __name__ == "__main__":
    import asyncio
    import logfire

    logfire.configure(send_to_logfire="if-token-present")
    logfire.instrument_pydantic_ai()

    out = asyncio.run(evaluate(conv_examples_test))
    analyze_predictions(out)  # prints all the slices, writes _joined.csv

    # If a DSPy predictions.csv exists for the same GEPA run, do the parity
    # comparison automatically. Same records → directly comparable accuracy.
    dspy_csv = Path("runs") / GEPA_NAME / "predictions.csv"
    if dspy_csv.exists():
        compare_runs(dspy_csv, out)
    else:
        print(
            f"\n(No {dspy_csv} found — skip parity report. Run "
            f"`RUN_GEPA=1 GEPA_NAME={GEPA_NAME} uv run python dspy_agent.py` "
            "first to enable side-by-side comparison.)"
        )
```

### Integration Points
```yaml
ENV VARS:
  - DEEPSEEK_API_KEY   (required, already used by dspy_agent.py)
  - GEPA_NAME          (optional, defaults to "gepa_smoke_20260429_204159")
  - LOGFIRE_TOKEN      (optional, enables remote tracing)

NO NEW DEPENDENCIES:
  - pydantic-ai, logfire, anthropic — all already in pyproject.toml

ARTIFACTS WRITTEN:
  - runs/<GEPA_NAME>/pydantic_predictions.csv
  - runs/<GEPA_NAME>/pydantic_predictions_joined.csv
  - (analyze_predictions writes the joined file with this same prefix because
    it derives the joined-path from the predictions-path's stem)
```

## List of Tasks

```yaml
Task 1: Skeleton + prompt loader
  CREATE pydantic_agent.py following the Module Skeleton block exactly:
    - Std-lib imports (asyncio, csv, json, os, Path, typing)
    - Third-party imports (dotenv, pydantic, pydantic_ai)
    - load_dotenv(Path.home() / ".env")  — MUST come BEFORE `from dspy_agent ...`
      because dspy_agent constructs dspy.LM at import time and reads
      DEEPSEEK_API_KEY then.
    - from dspy_agent import (Document, QAPair, ConversationHistory,
                              CALCULATOR_TOOLS, _DOCS, qa_data,
                              conv_examples_test, numeric_match,
                              analyze_predictions)
      Note `_DOCS` is required (used by ConversationRunner.run_conversation).
    - GEPA_NAME / PROMPTS_PATH / TurnType / ConvType
    - _load_optimized_prompts() — instructions-only, no field guidance
    - PROMPTS = _load_optimized_prompts(PROMPTS_PATH)
  VALIDATE:
    uv run python -c "
from pydantic_agent import PROMPTS, _DOCS, conv_examples_test, qa_data
assert set(PROMPTS) == {'triage','preprocess','retriever','calculator'}
for k, v in PROMPTS.items():
    assert isinstance(v, str) and len(v) > 50, k
assert len(_DOCS) > 0, 'document store imported empty'
assert len(conv_examples_test) == 100, f'expected 100 test convs, got {len(conv_examples_test)}'
print('prompts loaded:', {k: len(v) for k,v in PROMPTS.items()})
print('shared symbols ok — _DOCS:', len(_DOCS), 'test convs:', len(conv_examples_test))
"

Task 2: Output models — match DSPy signature outputs field-by-field
  ADD to pydantic_agent.py:
    - TriageOut(reasoning, turn_type, conv_type)         — mirrors TriageSignature outputs (+ ChainOfThought reasoning)
    - PreprocessOut(reasoning, sub_questions, program)   — mirrors PreprocessSignature outputs (+ ChainOfThought reasoning)
    - RetrievedValues(reasoning, answers)                — mirrors RetrieverSignature outputs (+ ChainOfThought reasoning)
    - CalcOut(answer)                                    — mirrors CalculationSignature outputs (NO reasoning; ReAct uses tools)
  VALIDATE:
    uv run python -c "
from pydantic_agent import TriageOut, PreprocessOut, RetrievedValues, CalcOut
from dspy_agent import QAPair
TriageOut(reasoning='because', turn_type='number', conv_type='Type I')
PreprocessOut(reasoning='because', sub_questions=['x'], program='add(A,B)')
RetrievedValues(reasoning='because', answers=[QAPair(question='q', answer='1.0')])
CalcOut(answer='3.14')
# Verify field-set parity vs DSPy signatures
assert set(TriageOut.model_fields) == {'reasoning','turn_type','conv_type'}
assert set(PreprocessOut.model_fields) == {'reasoning','sub_questions','program'}
assert set(RetrievedValues.model_fields) == {'reasoning','answers'}
assert set(CalcOut.model_fields) == {'answer'}
print('models ok — fields match DSPy signatures')
"

Task 3: Agents (no calculator tools yet)
  ADD to pydantic_agent.py:
    - LM_MINI (DeepSeek via OpenAIModel)
    - triage_agent / preprocess_agent / retriever_agent / calculator_agent
    - Each with output_type and system_prompt=PROMPTS[stage]
  VALIDATE:
    uv run python -c "
from pydantic_agent import triage_agent, preprocess_agent, retriever_agent, calculator_agent
for a in (triage_agent, preprocess_agent, retriever_agent, calculator_agent):
    assert a.system_prompt or a._system_prompts
print('agents constructed')
"

Task 4: Register calculator tools
  ADD to pydantic_agent.py:
    - Loop over CALCULATOR_TOOLS, call calculator_agent.tool_plain(fn)
  VALIDATE:
    uv run python -c "
from pydantic_agent import calculator_agent
# Pydantic AI exposes registered tools — count should be 6
n = len(calculator_agent._function_tools) if hasattr(calculator_agent, '_function_tools') else None
print('calc tools:', n)
"

Task 5: ConversationRunner
  ADD to pydantic_agent.py:
    - ConversationRunner class with _run_turn / run_conversation
    - Number path: triage -> retriever (single question)
    - Program path: triage -> preprocess -> retriever -> calculator
    - Threads ConversationHistory across turns within one conversation
  VALIDATE:
    uv run python -c "
import asyncio
from pydantic_agent import ConversationRunner
from dspy_agent import conv_examples_test
ex = conv_examples_test[0]
preds = asyncio.run(ConversationRunner().run_conversation(ex.report_id, ex.questions[:1]))
print('one-turn run:', preds)
"

Task 6: Evaluation harness + main
  ADD to pydantic_agent.py:
    - evaluate(examples) writes pydantic_predictions.csv, prints overall acc
    - __main__: configure logfire (if-token-present), run evaluate, then
      analyze_predictions on the resulting path
  VALIDATE:
    uv run python pydantic_agent.py
    # Expect: progress lines, overall accuracy, slice breakdowns,
    # files in runs/gepa_smoke_20260429_204159/

Task 7: Tests
  CREATE tests/test_pydantic_agent.py:
    - test_prompts_loaded_from_artifact: PROMPTS keys + non-empty
    - test_prompts_match_artifact_instructions: assert
        PROMPTS["triage"].startswith(raw_json["triage.predict"]["signature"]["instructions"][:50])
        for each stage (verifies we loaded from the right file)
    - test_output_models_validate: smoke construction of each model
    - test_calculator_has_six_tools: calculator_agent has 6 tools registered
    - test_runner_single_turn_number_question (uses TestModel override):
        triage returns turn_type=number, retriever returns one QAPair, runner
        returns its answer string. No real LM calls.
    - test_runner_single_turn_program_question (uses TestModel override):
        full 4-stage path, asserts final answer string
    - test_history_serialization_format: verify the user-message string
        passed to retriever contains the same `Q1 [report=...]:`/`A1:` format
        as ConversationHistory.as_text() (so the optimized prompt sees what
        it was optimized against)
    - test_output_models_match_dspy_signatures: import the four DSPy
        signatures and assert each Pydantic output_type's field set equals
        the DSPy signature's output_fields set, plus 'reasoning' for the three
        ChainOfThought ones:
            from dspy_agent import (
                TriageSignature, PreprocessSignature,
                RetrieverSignature, CalculationSignature,
            )
            # dspy.Signature exposes output_fields/input_fields as public dicts
            assert set(TriageOut.model_fields)       == set(TriageSignature.output_fields)       | {'reasoning'}
            assert set(PreprocessOut.model_fields)   == set(PreprocessSignature.output_fields)   | {'reasoning'}
            assert set(RetrievedValues.model_fields) == set(RetrieverSignature.output_fields)    | {'reasoning'}
            assert set(CalcOut.model_fields)         == set(CalculationSignature.output_fields)  # no reasoning for ReAct
    - test_render_chat_inputs_format: feed a known dict to _render_chat_inputs
        and assert the output is exactly:
            "[[ ## question ## ]]\nfoo\n[[ ## history ## ]]\n(no prior turns)\n[[ ## conv_type ## ]]\nType I"
        — preserves DSPy ChatAdapter wire format and field order
    - test_render_chat_inputs_field_order_matches_dspy_inputs: for each stage,
        capture the kwargs dict the runner builds and assert its key order
        matches the DSPy signature's InputField declaration order:
            triage:     ['question']
            preprocess: ['question','history','conv_type']
            retriever:  ['turn_type','questions','document','history']
            calculator: ['question','retrieved','program']
    - test_runner_uses_dspy_field_names: spy on each agent's .run() call (via
        TestModel + capture) and assert the user message contains
        `[[ ## <field> ## ]]` for every DSPy input field name — never a paraphrase
    - test_test_set_is_imported_from_dspy: verify pydantic_agent uses the
        SAME conv_examples_test object as dspy_agent (not a re-derived copy):
            from pydantic_agent import conv_examples_test as ct_pyd
            from dspy_agent  import conv_examples_test as ct_dspy
            assert ct_pyd is ct_dspy
            ids_pyd = [(ex.report_id, len(ex.questions)) for ex in ct_pyd]
            ids_dspy = [(ex.report_id, len(ex.questions)) for ex in ct_dspy]
            assert ids_pyd == ids_dspy
            assert len(ct_pyd) == 100  # current expected size
    - test_compare_runs_detects_drift: write two tiny synthetic CSVs that
        differ on one (report_id, turn_index) row and assert compare_runs
        raises RuntimeError mentioning "Test-set drift"
  VALIDATE:
    uv run pytest tests/test_pydantic_agent.py -v

Task 8: Lint + format
  RUN: ruff check pydantic_agent.py tests/test_pydantic_agent.py --fix
  RUN: ruff format pydantic_agent.py tests/test_pydantic_agent.py

Task 9: Same-records evaluation against dspy_agent.py
  PRECONDITION:
    - pydantic_agent.py imports `conv_examples_test` directly from dspy_agent
      (no independent test-set construction). Test set covered by Task 7's
      `test_test_set_is_imported_from_dspy`.
    - GEPA_NAME is identical for both runs (so the same optimized prompts are
      loaded and the same predictions.csv path is written/read).

  STEP 1 — Generate the DSPy reference predictions on the SAME records:
    RUN: RUN_GEPA=1 GEPA_NAME=gepa_smoke_20260429_204159 uv run python dspy_agent.py \
            | tee runs/gepa_smoke_20260429_204159/dspy_eval.log
    PRODUCES: runs/gepa_smoke_20260429_204159/predictions.csv
              (one row per (report_id, turn_index), same 100 conversations)

  STEP 2 — Run the Pydantic AI port on the SAME records:
    RUN: uv run python pydantic_agent.py \
            | tee runs/gepa_smoke_20260429_204159/pydantic_eval.log
    PRODUCES: runs/gepa_smoke_20260429_204159/pydantic_predictions.csv
              runs/gepa_smoke_20260429_204159/parity_report.csv  (auto from compare_runs)

  STEP 3 — Read the auto-generated parity report from the log:
    The pydantic_agent.py __main__ calls compare_runs() which prints:
      - n turns, dspy acc, pyd acc, delta_pp, agreement %
      - Counts: both correct / both wrong / only dspy / only pyd
      - Per-slice deltas: turn_type, conv_type, q_order
    AND writes parity_report.csv (one row per turn, both predictions side-by-side).

  STEP 4 — Apply the gates:
    GATE A (record coverage): compare_runs must NOT raise. If it raises with
      "Test-set drift", pydantic_agent is not running on the same records as
      dspy_agent — fix the import (likely re-derived conv_examples_test) and
      re-run.
    GATE B (overall parity): |pyd_acc - dspy_acc| <= 5 pp. If wider, port is
      not behavior-preserving.
    GATE C (slice parity): for every (turn_type, conv_type, q_order) slice
      with n >= 5, |delta_pp| <= 10. If any slice exceeds, INVESTIGATE
      before proceeding — likely causes:
        - History serialization format diverged from ConversationHistory.as_text()
        - Document JSON layout differs from Document.model_dump_json(indent=2)
        - DSPy ChatAdapter `[[ ## field ## ]]` markers vs Pydantic AI typed
          output cause the LLM to interpret "answer" differently
        - DeepSeek reasoning_content handling differs between LiteLLM and the
          OpenAI-compatible client

  STEP 5 — Triage disagreements (optional but recommended):
    Open parity_report.csv, filter to rows where agree==False, look for
    patterns: same turn_type? same q_order? same kind of answer (percentage
    vs raw number)? Surface findings in a short note appended to
    pydantic_eval.log.
```

## Validation Loop

### Level 1: Syntax & Style
```bash
ruff check pydantic_agent.py --fix
ruff format pydantic_agent.py
ruff check tests/test_pydantic_agent.py --fix
ruff format tests/test_pydantic_agent.py
```

### Level 2: Unit Tests
```bash
uv run pytest tests/test_pydantic_agent.py -v -x
```

### Level 3: Smoke Test
```bash
uv run python -c "
from pydantic_agent import PROMPTS, ConversationRunner
from dspy_agent import conv_examples_test
import asyncio
print({k: len(v) for k, v in PROMPTS.items()})
ex = conv_examples_test[0]
print('test conv:', ex.report_id, len(ex.questions), 'turns')
preds = asyncio.run(ConversationRunner().run_conversation(ex.report_id, ex.questions))
print('preds:', preds)
print('gold :', ex.gold_answers)
"
```

### Level 4: Full Eval + Comparison
```bash
# Pydantic AI run
uv run python pydantic_agent.py

# DSPy run loading the same optimized_runner.json
RUN_GEPA=1 GEPA_NAME=gepa_smoke_20260429_204159 uv run python dspy_agent.py

# Diff per-turn predictions to find any systematic divergence
python3 -c "
import pandas as pd
d = pd.read_csv('runs/gepa_smoke_20260429_204159/predictions.csv').rename(columns={'pred_answer':'pred_dspy','correct':'corr_dspy'})
p = pd.read_csv('runs/gepa_smoke_20260429_204159/pydantic_predictions.csv').rename(columns={'pred_answer':'pred_pyd','correct':'corr_pyd'})
m = d.merge(p, on=['report_id','turn_index','question','gold_answer'])
print('agree:', (m.corr_dspy == m.corr_pyd).mean())
print('only dspy correct:', ((m.corr_dspy) & (~m.corr_pyd)).sum())
print('only pyd correct :', ((~m.corr_dspy) & (m.corr_pyd)).sum())
"
```

## Final Validation Checklist
- [ ] `uv run python pydantic_agent.py` runs end-to-end
- [ ] Overall turn accuracy within ±5 pp of `dspy_agent.py` loaded from the same artifact
- [ ] All four `system_prompt` strings exactly match `signature.instructions` from `optimized_runner.json` (modulo the appended Field guidance section)
- [ ] Calculator agent has the six calc tools registered
- [ ] `pytest tests/test_pydantic_agent.py -v` all green
- [ ] `ruff check` zero errors, `ruff format --check` no diffs
- [ ] `pydantic_predictions.csv` and `pydantic_predictions_joined.csv` written
- [ ] Slice analysis (`turn_type`, `conv_type`, `q_order`, plus 2-way pivots) printed
- [ ] No new dependencies added to `pyproject.toml`
- [ ] `GEPA_NAME` env var override works (try a different run dir and confirm load)

---

## Anti-Patterns to Avoid
- Do not drop `reasoning` from TriageOut / PreprocessOut / RetrievedValues — DSPy ChainOfThought adds it and the optimized prompts elicit it. Dropping it changes elicitation behavior even if you keep the same instructions.
- Do not add a `reasoning` field to CalcOut — `dspy.ReAct` does not have one; the trajectory IS the reasoning, captured as tool-call messages by Pydantic AI.
- Do not rename DSPy input field names in the wire format. `questions` (not `lookup_questions`), `retrieved` (not `values`), `document` (not `doc`), `turn_type` (not `mode`). Renaming silently breaks any prompt fragment that references the field by name.
- Do not use prose labels (`Question:`, `Doc:`) in user messages — the optimized prompts expect DSPy ChatAdapter `[[ ## name ## ]]` markers. Use `_render_chat_inputs()`.
- Do not modify `dspy_agent.py` — port, don't rewrite. Import shared bits.
- Do not re-implement the calculator tools — import `CALCULATOR_TOOLS`.
- Do not re-implement the test set construction — import `conv_examples_test`.
- Do not re-implement `analyze_predictions` — import and reuse it.
- Do not load `calculator.extract.predict` from the JSON — Pydantic AI extracts typed output natively.
- Do not change the `ConversationHistory.as_text()` format — the prompts were optimized against it.
- Do not switch to Pydantic AI's `message_history` parameter for prior turns — same reason.
- Do not call sub-agents in parallel within a turn — they're a sequential pipeline.
- Do not parallelize across *turns* of a conversation — turns depend on prior history.
- Do not add a retry/timeout layer before measuring baseline parity. Layer those in *after* the port is proven equivalent, in a follow-up.
- Do not hardcode the prompts file path — read `GEPA_NAME` from env so the same script can re-evaluate any GEPA run.
- Do not silently fall back to a different model. If `DEEPSEEK_API_KEY` isn't set, fail fast with a clear error.

## Confidence Score: 8/10

**Strengths:**
- Reuses every framework-agnostic piece from `dspy_agent.py` (models, tools, test set, analysis), minimizing surface area
- The four optimized prompts are loaded verbatim — no re-engineering, no re-optimization, just a runtime swap
- Validation includes a direct DSPy-vs-Pydantic-AI per-turn diff, so any port regression is immediately diagnosable
- No new dependencies — all pinned already
- The orchestration shape (sequential, per-turn, history-threaded) is identical to the reference, so behavior parity is the default outcome

**Risks:**
- Pydantic AI's structured-output mechanism asks the model differently than DSPy's ChatAdapter (`[[ ## field ## ]]` markers vs. JSON schema). Even with identical instructions, the *output format* the LLM produces differs — this could measurably move accuracy in either direction. The DSPy-vs-Pydantic-AI diff in Level 4 will reveal this.
- DeepSeek via OpenAI-compatible endpoint may behave slightly differently for structured output than via DSPy's LiteLLM path (e.g. handling of `reasoning_content`). If accuracy diverges, switching to native Anthropic models is a viable fallback at the cost of breaking model parity with the GEPA optimization.
- The `calculator.react` instructions reference ReAct-style trajectories. Pydantic AI's tool-call loop is functionally equivalent but doesn't print thought/action/observation strings. The instructions still work, but the prompt's mental model of "you produce a trajectory" may need a small follow-up edit if the calculator agent underperforms.
